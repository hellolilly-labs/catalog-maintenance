"""
Pinecone Vector Database Setup for Multi-Namespace RAG System

This script handles the setup and data ingestion for a retail catalog RAG (Retrieval-Augmented Generation)
system using Pinecone vector database. It processes product catalogs, informational content, and brand
information from various file formats and organizes them into separate namespaces for optimized retrieval.

Usage:
    python pinecone_setup.py

Environment variables required:
    PINECONE_API_KEY - Your Pinecone API key
"""

import os
import re
import json
import time
import uuid
import logging
import sys
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI
import argparse

# Install required packages if not present
try:
    from tqdm import tqdm
    import PyPDF2  # Import PyPDF2 for PDF processing
    import PyCryptodome
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "PyPDF2", "PyCryptodome"])
    from tqdm import tqdm
    import PyPDF2

# Add the path to the parent directory containing spence
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
spence_dir = os.path.join(parent_dir, "spence")
if (spence_dir not in sys.path):
    sys.path.insert(0, spence_dir)

from spence.rag import PineconeRAG


# Initialize logging with a format that includes timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Import Pinecone client
from pinecone import Pinecone, PodSpec, SearchQuery, SearchRerank, RerankModel

# Configuration
# INDEX_NAME = "specialized-detailed"
# INDEX_NAME = "sundayriley-llama-2048"
# INDEX_NAME = "balenciaga-llama-2048"
INDEX_NAME = "gucci-llama-2048"
# DATA_DIRECTORY = "./rag/data/specialized.com/"
# DATA_DIRECTORY = "./rag/data/balenciaga.com/"
DATA_DIRECTORY = "./rag/data/gucci.com/"
# INDEX_NAME = "specialized-llama-2048"
# DATA_DIRECTORY = "./data/"
EXPORT_DIRECTORY = "./export/"
# Change embedding model to use Pinecone's integrated model
EMBEDDING_MODEL = "llama-text-embed-v2"  # Pinecone integrated embedding model
IMPORT_DATA = True
BATCH_SIZE = 45
VECTOR_DIMENSION = 2048  # llama-text-embed-v2 with 2048 dimensions for better quality
# VECTOR_DIMENSION = 1024  # llama-text-embed-v2 with 2048 dimensions for better quality
BATCH_SLEEP_TIME = 5  # Seconds to sleep between batches

# Available namespaces
NAMESPACES = ["products", "information", "brand"]

# Initialize Pinecone client
try:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    logger.info("Pinecone client initialized successfully")
except KeyError:
    logger.error("PINECONE_API_KEY not found in environment variables")
    raise EnvironmentError("PINECONE_API_KEY not found. Please set up your .env file.")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone client: {e}")
    raise

# Fancy display
print("\n============================================================")
print("====== PINECONE RAG SETUP - MULTI-NAMESPACE INGESTION ======")

# Fancy display
print("\n============================================================")
print("====== PINECONE RAG SETUP - MULTI-NAMESPACE INGESTION ======")
print("============================================================\n")

print(f"🏗️  Initializing setup for index: {INDEX_NAME}")

# =========== UTILITY FUNCTIONS ===========

def extract_price_range(price: str) -> str:
    """
    Categorize a product price into a price range bucket.
    
    Args:
        price (str): Price as a string, possibly with currency symbols
        
    Returns:
        str: Price range category (budget, mid-range, premium, high-end, or unknown)
    """
    if not price:
        return "unknown"
        
    price_num = re.sub(r'[^\d.]', '', price)
    try:
        price_value = float(price_num)
        if price_value <= 500:
            return "budget"
        elif price_value <= 2000:
            return "mid-range"
        elif price_value <= 5000:
            return "premium"
        else:
            return "high-end"
    except ValueError:
        return "unknown"

def extract_product_features(product: Dict[str, Any]) -> List[str]:
    """
    Extract key product features based on product category and specifications.
    
    Args:
        product (dict): Product data dictionary
        
    Returns:
        list: List of key product features as strings
    """
    features = []
    categories = product.get('categories', [])
    specifications = product.get('specifications', {})
    
    # Use sale price if available, otherwise use original price
    if price := (product.get('salePrice') or product.get('originalPrice')):
        features.append(f"Price: {price}")
    
    # Add sizes information
    if sizes := product.get('sizes', []):
        features.append(f"Available sizes: {', '.join(sizes)}")
    
    if colors := product.get('colors', []):
        features.append(f"Available in: {', '.join(colors)}")
    
    # Get main category
    main_category = categories[0].lower() if categories else ""
    
    # Extract all specifications
    if specifications:
        for spec_category, spec_details in specifications.items():
            if isinstance(spec_details, dict):
                for spec_name, spec_value in spec_details.items():
                    if isinstance(spec_value, list):
                        spec_value = ', '.join(spec_value)
                    features.append(f"{spec_category} - {spec_name}: {spec_value}")
            elif isinstance(spec_details, list):
                spec_value = ', '.join(spec_details)
                features.append(f"{spec_category}: {spec_value}")
            else:
                features.append(f"{spec_category}: {spec_details}")
    
    # Extract category-specific features
    if main_category == "bikes" or "bikes" in [c.lower() for c in categories]:
        # Extract bike-specific features
        if "Frame" in specifications:
            if "frame_material" in specifications["Frame"]:
                features.append(f"Frame: {specifications['Frame']['frame_material']}")
            if "geometry" in specifications["Frame"]:
                features.append(f"Geometry: {specifications['Frame']['geometry']}")
        
        if "Suspension" in specifications:
            if "travel" in specifications["Suspension"]:
                features.append(f"Suspension travel: {specifications['Suspension']['travel']}")
            if "fork" in specifications["Suspension"]:
                features.append(f"Fork: {specifications['Suspension']['fork']}")
            if "rear_shock" in specifications["Suspension"]:
                features.append(f"Rear shock: {specifications['Suspension']['rear_shock']}")
        
        if "Drivetrain" in specifications:
            if "groupset" in specifications["Drivetrain"]:
                features.append(f"Groupset: {specifications['Drivetrain']['groupset']}")
            if "crankset" in specifications["Drivetrain"]:
                features.append(f"Crankset: {specifications['Drivetrain']['crankset']}")
        
        # Check for electric bikes
        if any("electric" in c.lower() or "e-bike" in c.lower() or "turbo" in c.lower() for c in categories):
            if "Motor" in specifications:
                if "type" in specifications["Motor"]:
                    features.append(f"Motor: {specifications['Motor']['type']}")
                if "max_power" in specifications["Motor"]:
                    features.append(f"Power: {specifications['Motor']['max_power']}")
            
            if "Battery" in specifications and "capacity" in specifications["Battery"]:
                features.append(f"Battery: {specifications['Battery']['capacity']}")
    
    elif "helmet" in main_category.lower() or any("helmet" in c.lower() for c in categories):
        if "Safety" in specifications and "certification" in specifications["Safety"]:
            features.append(f"Certification: {specifications['Safety']['certification']}")
        
        if "Features" in specifications:
            if "ventilation" in specifications["Features"]:
                features.append(f"Ventilation: {specifications['Features']['ventilation']}")
            if "mips" in specifications["Features"]:
                features.append(f"MIPS: {specifications['Features']['mips']}")
    
    elif "shoe" in main_category.lower() or any("shoe" in c.lower() for c in categories):
        if "Construction" in specifications:
            if "closure" in specifications["Construction"]:
                features.append(f"Closure: {specifications['Construction']['closure']}")
            if "sole" in specifications["Construction"]:
                features.append(f"Sole: {specifications['Construction']['sole']}")
    
    # Extract key selling points from the descriptor (or description)
    if descriptor := product.get('descriptor', ''):
        # Extract first 2 sentences for key selling points
        sentences = re.split(r'(?<=[.!?])\s+', descriptor)[:2]
        short_desc = ' '.join(sentences)
        if short_desc:
            features.append(f"Key Selling Points: {short_desc}")
    # If no descriptor, use description
    elif description := product.get('description', ''):
        # Extract first 2 sentences for key selling points
        sentences = re.split(r'(?<=[.!?])\s+', description)[:2]
        short_desc = ' '.join(sentences)
        if short_desc:
            features.append(f"Description: {short_desc}")
    
    return features

def determine_namespace(file_path: str, content_type: Optional[str] = None) -> str:
    """
    Determine which namespace to use based on filename and content.
    
    Args:
        file_path (str): Path to the data file
        content_type (str, optional): Content type if known
        
    Returns:
        str: Target namespace for this content
    """
    filename = os.path.basename(file_path).lower()
    
    # Check filename first for clearest signals
    if "product" in filename:
        return "products"
    elif "brand" in filename:
        return "brand"
    
    # If content_type is explicitly product, use products namespace
    if content_type == "product":
        return "products"
        
    # Default to information for all other content
    return "information"

# Add these utility functions for text chunking

def estimate_tokens(text: str) -> int:
    """
    Roughly estimate the number of tokens in text.
    OpenAI models use ~4 chars per token for English text.
    
    Args:
        text (str): Text to estimate
    
    Returns:
        int: Estimated token count
    """
    return len(text) // 4  # Simple approximation: 1 token ≈ 4 characters

def chunk_text(text: str, max_tokens: int = 8000, document_type: str = "text", chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split text into chunks that don't exceed token limits with improvements for different document types.
    
    Args:
        text (str): Text to chunk
        max_tokens (int): Maximum tokens per chunk (default: 8000)
        document_type (str): Type of document ("text", "pdf", "data")
        chunk_overlap (int): Number of tokens to overlap between chunks
        
    Returns:
        List[Dict[str, Any]]: List of chunk objects with text and metadata
    """
    # If text is already within limits, return as is
    if estimate_tokens(text) <= max_tokens:
        return [{"text": text, "metadata": {"is_chunk": False}}]
    
    chunks = []
    
    # Document type-specific preprocessing
    if document_type == "pdf":
        # For PDFs: Try to identify and respect page boundaries
        pages = re.split(r'---\s*Page\s+\d+\s*---', text)
        
        if len(pages) > 1:
            # Process each page separately with potential chunking
            for i, page_text in enumerate(pages):
                if not page_text.strip():
                    continue
                    
                page_num = i + 1  # Approximate page number
                # Mark page boundaries for better context
                page_text = f"[Page {page_num}]\n{page_text.strip()}"
                
                if estimate_tokens(page_text) <= max_tokens:
                    chunks.append({
                        "text": page_text,
                        "metadata": {
                            "is_chunk": False,
                            "page_number": page_num
                        }
                    })
                else:
                    # If page is too large, chunk it
                    page_chunks = _chunk_by_structure(page_text, max_tokens, chunk_overlap)
                    for j, chunk_text in enumerate(page_chunks):
                        chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                "is_chunk": True,
                                "page_number": page_num,
                                "chunk_number": j + 1,
                                "total_chunks_in_page": len(page_chunks)
                            }
                        })
            
            # Return page-aware chunks
            return chunks
    
    elif document_type == "data":
        # For data: try to respect data record boundaries
        # This is a simplified approach - real data parsing would be more complex
        
        # Check for JSON-like or tabular structure
        if text.strip().startswith('{') or text.strip().startswith('['):
            # Likely JSON data, try to parse records
            try:
                # For JSON arrays, split by records
                if text.strip().startswith('['):
                    import json
                    data = json.loads(text)
                    if isinstance(data, list) and len(data) > 1:
                        record_chunks = []
                        current_chunk = []
                        current_tokens = 0
                        
                        for record in data:
                            record_text = json.dumps(record, indent=2)
                            record_tokens = estimate_tokens(record_text)
                            
                            if record_tokens > max_tokens:
                                # Handle large individual records
                                record_chunks.append(record_text)
                            elif current_tokens + record_tokens > max_tokens:
                                # Start a new chunk
                                record_chunks.append(json.dumps(current_chunk, indent=2))
                                current_chunk = [record]
                                current_tokens = record_tokens
                            else:
                                # Add to current chunk
                                current_chunk.append(record)
                                current_tokens += record_tokens
                        
                        # Add the final chunk if not empty
                        if current_chunk:
                            record_chunks.append(json.dumps(current_chunk, indent=2))
                        
                        # Convert to our chunk format
                        for i, chunk_text in enumerate(record_chunks):
                            chunks.append({
                                "text": chunk_text,
                                "metadata": {
                                    "is_chunk": True,
                                    "data_format": "json",
                                    "chunk_number": i + 1,
                                    "total_chunks": len(record_chunks)
                                }
                            })
                        return chunks
            except:
                pass  # Fall back to regular chunking
    
    # Default: use the improved structural chunking with overlap
    chunk_texts = _chunk_by_structure(text, max_tokens, chunk_overlap)
    
    # Convert to chunk objects with metadata
    for i, chunk_text in enumerate(chunk_texts):
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "is_chunk": True,
                "chunk_number": i + 1,
                "total_chunks": len(chunk_texts)
            }
        })
    
    return chunks

def _chunk_by_structure(text: str, max_tokens: int, overlap_tokens: int, recursion_level: int = 0) -> List[str]:
    """
    Helper function that implements improved structural chunking with overlap.
    
    Args:
        text: The text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap
        recursion_level: Current recursion depth to prevent stack overflow
        
    Returns:
        List of text chunks
    """
    # Safety check to prevent infinite recursion
    if recursion_level > 10:
        # If we've recursed too deep, fall back to simple chunking by paragraphs
        logger.warning("Maximum recursion depth reached, falling back to simple chunking")
        # Simple fallback chunking - just divide by paragraphs with no further recursion
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in re.split(r'\n\s*\n', text):
            paragraph_tokens = estimate_tokens(paragraph)
            
            if current_tokens + paragraph_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
                current_tokens = paragraph_tokens
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += paragraph_tokens
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    # Ensure we're not trying to process empty text or text that's too small to need chunking
    if not text or len(text.strip()) < 10:
        return [text] if text else []
    
    chunks = []
    # Check for section headers (both Markdown and structural headers)
    section_pattern = r'(?:^|\n)(#+\s+.+|\d+\.\s+[A-Z].+|CHAPTER\s+[\dIVXLC]+|[A-Z][A-Z\s]+\n)'
    sections = re.split(section_pattern, text, flags=re.MULTILINE)
    
    # If we found clear sections
    if len(sections) > 2:  # More than just a split of the first section
        # Process each section with potential sub-chunking
        current_chunk = ""
        current_tokens = 0
        section_header = ""
        last_overlap = ""
        
        for i, section in enumerate(sections):
            # If this is a header
            if i % 2 == 1:
                section_header = section
                continue
                
            # If we have content to process
            if section_header:
                # Combine header with content
                full_section = f"{section_header}\n{section}"
                section_tokens = estimate_tokens(full_section)
                
                # If adding this section would exceed the limit
                if current_tokens + section_tokens > max_tokens:
                    # Start a new chunk
                    if current_chunk:
                        chunks.append(current_chunk)
                        last_overlap = _extract_overlap(current_chunk, overlap_tokens)
                    
                    # If the section itself is too large, recursively chunk it
                    if section_tokens > max_tokens:
                        # Verify the section is actually getting smaller to prevent infinite recursion
                        if len(full_section.strip()) < len(text.strip()) * 0.9:  # Section is at least 10% smaller
                            sub_chunks = _chunk_by_structure(full_section, max_tokens, overlap_tokens, recursion_level + 1)
                            chunks.extend(sub_chunks)
                        else:
                            # If section isn't getting smaller, force-split by paragraphs
                            logger.warning("Section not getting smaller, forcing paragraph split")
                            for para in re.split(r'\n\s*\n', full_section):
                                if estimate_tokens(para) > max_tokens:
                                    # Further split by sentences if needed
                                    for sent in re.split(r'(?<=[.!?])\s+', para):
                                        if sent.strip():
                                            chunks.append(sent)
                                else:
                                    chunks.append(para)
                        current_chunk = ""
                        current_tokens = 0
                    else:
                        # Start a new chunk with this section and include overlap
                        current_chunk = last_overlap + full_section if last_overlap else full_section
                        current_tokens = estimate_tokens(current_chunk)
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + full_section
                    else:
                        current_chunk = full_section
                    current_tokens += section_tokens
                
                section_header = ""
            elif section.strip():
                # Handle text before any headers (prologue)
                prologue_tokens = estimate_tokens(section)
                
                if prologue_tokens > max_tokens:
                    # Check for potential infinite recursion
                    if len(section.strip()) < len(text.strip()) * 0.9:
                        # Recursively process this content
                        sub_chunks = _chunk_by_structure(section, max_tokens, overlap_tokens, recursion_level + 1)
                        chunks.extend(sub_chunks)
                    else:
                        # Fall back to simple splitting
                        logger.warning("Prologue not getting smaller, falling back to simple chunking")
                        simple_chunks = [p for p in re.split(r'\n\s*\n', section) if p.strip()]
                        chunks.extend(simple_chunks)
                else:
                    current_chunk = section
                    current_tokens = prologue_tokens
        
        # Add the final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
    else:
        # No clear sections, try paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = ""
        current_tokens = 0
        last_overlap = ""
        
        for paragraph in paragraphs:
            paragraph_tokens = estimate_tokens(paragraph)
            
            # If paragraph itself exceeds limit, split by sentences
            if paragraph_tokens > max_tokens:
                # Process oversized paragraph by sentences
                if current_chunk:
                    chunks.append(current_chunk)
                    last_overlap = _extract_overlap(current_chunk, overlap_tokens)
                    current_chunk = last_overlap
                    current_tokens = estimate_tokens(last_overlap)
                
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    sentence_tokens = estimate_tokens(sentence)
                    
                    if current_tokens + sentence_tokens > max_tokens:
                        # Start a new chunk with overlap
                        if current_chunk:
                            chunks.append(current_chunk)
                            last_overlap = _extract_overlap(current_chunk, overlap_tokens)
                            current_chunk = last_overlap + sentence
                            current_tokens = estimate_tokens(current_chunk)
                        else:
                            current_chunk = sentence
                            current_tokens = sentence_tokens
                    else:
                        # Add to current chunk
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                        current_tokens += sentence_tokens
            # Normal sized paragraph
            elif current_tokens + paragraph_tokens > max_tokens:
                # Start a new chunk with overlap
                chunks.append(current_chunk)
                last_overlap = _extract_overlap(current_chunk, overlap_tokens)
                current_chunk = last_overlap + "\n\n" + paragraph
                current_tokens = estimate_tokens(current_chunk)
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += paragraph_tokens
        
        # Add the final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
    
    return chunks

def _extract_overlap(text: str, overlap_tokens: int) -> str:
    """
    Extract the end portion of text to use as overlap in the next chunk.
    
    Args:
        text: The source text
        overlap_tokens: Approximate number of tokens to include
        
    Returns:
        String with the overlap text
    """
    # Roughly estimate how many characters for the overlap
    overlap_chars = overlap_tokens * 4  # Rough approximation
    
    if len(text) <= overlap_chars:
        return text
    
    # Try to find a sentence boundary for cleaner overlap
    overlap_text = text[-overlap_chars:]
    sentence_match = re.search(r'[.!?]\s+[A-Z]', overlap_text)
    
    if sentence_match:
        # Found a sentence boundary in the overlap, start from there
        start_pos = sentence_match.end() - 1  # Include the capital letter
        return overlap_text[start_pos:]
    else:
        # No clear sentence boundary, try to find a paragraph or reasonable break
        para_match = re.search(r'\n\s*\n', overlap_text)
        if para_match:
            return overlap_text[para_match.end():]
        
        # Fall back to word boundary
        word_match = re.search(r'\s+[^\s]+$', overlap_text[:overlap_chars//2])
        if word_match:
            return overlap_text[word_match.start()+1:]
    
    # Last resort - just use the characters
    return overlap_text

# =========== DATA PROCESSING FUNCTIONS ===========

def process_product_data(products):
    chunks = []
    
    for i, product in enumerate(products):
        try:
            # category is the categories array joined by commas
            categories = product.get('categories', ['general'])
            if not categories:
                categories = ["general"]
            category = ", ".join(categories)
            
            # price is the salePrice if available, otherwise the originalPrice
            price = product.get('salePrice') or product.get('originalPrice') or "Unknown"
            
            # specifications is a dictionary of product specifications
            # e.g. {"Frame": {"frame_material": "Carbon", "geometry": "Endurance"}}
            specifications = product.get('specifications')
            if not specifications:
                specifications_list = product.get('additional_details', [])
                specifications = {}
                for spec in specifications_list:
                    if isinstance(spec, dict):
                        for k, v in spec.items():
                            if isinstance(v, list):
                                specifications[k] = ', '.join(v)
                            else:
                                specifications[k] = v
                    elif isinstance(spec, str):
                        # If it's a string, split by colon and add to specifications
                        index = spec.find(':')
                        if index == -1:
                            specifications["Spec"] = spec
                        else:
                            key = spec[:index].strip()
                            value = spec[index + 1:].trip()
                            specifications[key] = value
            
            specifications = specifications or {}
            
            # convert specifications to a markdown string for embedding
            specifications_md = "\n".join([f"- **{k}**: {v}" for k, v in specifications.items()])
            # log the specifications for the first product
            if i == 0:
                logger.info(f"Product specifications: {specifications}")
            
            # Construct a rich product description optimized for semantic search
            descriptor = None
            name = product.get('name') or product.get('title') or 'Unknown'
            brand = product.get('brand') or None
            product_text = f"""Product: {name}"""
            if brand:
                product_text += f" by {brand}"
            product_text += "\n"
            if category:
                product_text += f"""Category: {category}\n"""
            if price:
                # price_range = extract_price_range(price)
                # product_text += f"""Price Range: {price_range}\n"""
                product_text += f"Price: {price}\n"

            colors = product.get('colors', [])
            if colors and len(colors) > 0:
                product_text += f"Colors:\n"
                for color in colors:
                    if isinstance(color, dict):
                        # If color is a dict, use the 'name' key
                        product_text += f"- {color.get('name', color.get('colorFamily', 'Unknown'))}\n"
                    else:
                        product_text += f"- {color}\n"
            sizes = product.get('sizes', [])
            if sizes and len(sizes) > 0:
                if isinstance(sizes[0], dict):
                    # If sizes are dicts, use the 'name' key
                    sizes = [size.get('name', size.get('size', 'Unknown')) for size in sizes]
                product_text += f"Sizes: {', '.join(sizes)}\n"
            
            product_text += f"""Description: {product.get('description', '')}\n"""
            
            # Add specifications to the product text
            if specifications_md:
                product_text += f"Specifications:\n{specifications_md}\n"

            # if product.get('descriptor'):
            #     descriptor = product.get('descriptor')
            # else:
            #     continue

            # if the product has a clause about what the product is perfect for, add it to the description
            use_clause = product.get('useClause', '')
            if use_clause:
                product_text += f"\n{use_clause}"

    # This is a {product.get('main_category', 'bike').lower()} that's perfect for {product.get('sub_category', '').lower()} riding.
    #         if "mountain" in product.get('main_category', '').lower() or "mountain" in product.get('sub_category', '').lower():
    #             # Add mountain bike specific terms for better matching
    #             product_text += " Perfect for trail riding, mountain biking, off-road adventures, MTB, rough terrain."
            
            # Create document - ONE CHUNK PER PRODUCT
            product_id = f"{product.get('id', f'product-{i}')}"
            chunk = {
                "id": product_id,
                "text": descriptor or product_text,
                "metadata": {
                    "id": product_id,
                    "categories": product.get('categories', ['general']),
                    "name": name,
                    "description": product.get('description', product.get('descriptor', '')),
                "brand": brand or "",
                    "original_price": product.get('originalPrice', 'Unknown'),
                    "sale_price": product.get('salePrice', 'Unknown'),
                    "product_url": product.get('productUrl', ''),
                    "image_urls": product.get('imageUrls', [])[0:3],
                    "video_urls": product.get('videoUrls', [])[0:3],
                    "sizes": sizes,
                    "colors": colors,
                    "specifications": specifications,
                    # "related_products": product.get('relatedProducts', []),
                    "content_type": "product"
                }
            }
            if not descriptor is None:
                chunk['alternate_text'] = product_text
            chunks.append(chunk)
        except Exception as e:
            logger.error(f"Error processing product {i}: {e}")
            continue
    
    return chunks

def process_information_data(data: Dict[str, Any], content_type: str = "information") -> List[Dict[str, Any]]:
    """
    Process structured information data with sections.
    
    Args:
        data (dict): Information data with sections
        content_type (str): Content type for metadata
        
    Returns:
        list: List of processed information chunks
    """
    chunks = []
    title = data.get("title", "Information")
    category = data.get("category", "general")
    
    # Process each section
    for i, section in enumerate(data.get("sections", [])):
        section_title = section.get("title", f"Section {i+1}")
        section_content = section.get("content", "")
        
        # Skip empty sections
        if not section_content:
            continue
            
        # Create a stable ID
        section_id = f"{category}-{title.lower().replace(' ', '-')}-{i}"
        
        # Create metadata
        metadata = {
            "title": f"{title} - {section_title}",
            "category": category,
            "content_type": content_type,
            "source": section_content[:300]
        }
        
        # Create the information chunk
        chunks.append({
            "id": section_id,
            "text": f"# {section_title}\n\n{section_content}",
            "metadata": metadata
        })
    
    logger.info(f"Created {len(chunks)} information chunks from sections")
    return chunks

def process_generic_json(data: Any, filename: str, content_type: str = "information") -> List[Dict[str, Any]]:
    """
    Process generic JSON data of any structure.
    
    Args:
        data: JSON data (dict, list, etc.)
        filename (str): Source filename
        content_type (str): Content type for metadata
        
    Returns:
        list: List of processed chunks
    """
    chunks = []
    base_id = filename.replace('.json', '').lower().replace(' ', '-')
    
    # Detect category from content
    category = "general"
    if "size" in filename.lower() or "sizing" in json.dumps(data)[:500].lower():
        category = "sizing_guide"
    elif "maintenance" in filename.lower() or "care" in json.dumps(data)[:500].lower():
        category = "maintenance" 
    
    # Process based on data structure
    if isinstance(data, dict):
        # Process dictionary as sections
        for key, value in data.items():
            section_id = f"{base_id}-{key.lower().replace(' ', '-')}"
            section_content = ""
            
            if isinstance(value, dict):
                # Format nested dictionary as subsections
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        section_content += f"### {subkey}\n\n"
                        for k, v in subvalue.items():
                            section_content += f"- {k}: {v}\n"
                    elif isinstance(subvalue, list):
                        section_content += f"### {subkey}\n\n"
                        for item in subvalue:
                            if isinstance(item, str):
                                section_content += f"- {item}\n"
                            else:
                                section_content += f"- {json.dumps(item)}\n"
                    else:
                        section_content += f"### {subkey}\n\n{subvalue}\n\n"
            
            elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                # Convert list of objects to statements
                for item in value:
                    for k, v in item.items():
                        section_content += f"- {k}: {v}\n"
            elif isinstance(value, list):
                # Simple list
                for item in value:
                    section_content += f"- {item}\n"
            else:
                # Simple value
                section_content = str(value)
            
            # Only create chunk if we have content
            if section_content:
                chunks.append({
                    "id": section_id,
                    "text": f"# {key}\n\n{section_content}",
                    "metadata": {
                        "title": key,
                        "category": category,
                        "content_type": content_type
                    }
                })
    
    elif isinstance(data, list):
        # For lists, create one chunk with all items
        content = ""
        
        # If list items are dictionaries, convert to statements
        if all(isinstance(item, dict) for item in data):
            for item in data:
                # Format as statements
                for k, v in item.items():
                    content += f"- {k}: {v}\n"
        else:
            # Simple list
            for item in data:
                content += f"- {item}\n"
        
        chunks.append({
            "id": base_id,
            "text": content,
            "metadata": {
                "title": base_id.replace('-', ' ').title(),
                "category": category,
                "content_type": content_type
            }
        })
    
    logger.info(f"Created {len(chunks)} chunks from generic JSON data")
    return chunks

# Update the process_markdown_content function

def process_markdown_content(content: str, filename: str, content_type: str = "information") -> List[Dict[str, Any]]:
    """Process markdown content into proper-sized chunks for embedding."""
    chunks = []
    doc_id = os.path.splitext(filename)[0].lower().replace(' ', '-')
    
    # Determine category without changing content_type
    category = "general"
    if "size" in filename.lower() or "sizing" in content.lower()[:500]:
        category = "sizing_guide"
    elif "maintenance" in filename.lower() or "care" in content.lower()[:500]:
        category = "maintenance"
    
    if "brand" in filename.lower() and "about" in filename.lower():
        content_type = "brand"
    
    # Check if content exceeds token limit and needs chunking
    estimated_tokens = estimate_tokens(content)
    if estimated_tokens > 8000:
        # Split into smaller chunks using our utility function
        try:
            chunked_texts = chunk_text(content)
            
            # Create a document for each chunk
            for i, chunk_content in enumerate(chunked_texts):
                chunk_id = f"{doc_id}-chunk-{i+1}"
                
                # For first chunk, use original title; for others add part number
                title = doc_id.replace('-', ' ').title()
                if i > 0:
                    title += f" (Part {i+1})"
                    
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_content,
                    "metadata": {
                        "title": title,
                        "category": category,
                        "content_type": content_type,
                        "chunk_number": i+1,
                        "total_chunks": len(chunked_texts)
                    }
                })
        except Exception as e:
            logger.error(f"Error chunking content: {e}")
            # Fallback: Simple splitting by paragraphs
            paragraphs = content.split("\n\n")
            for i, para in enumerate(paragraphs[:10]):  # Limit to first 10 paragraphs if very large
                if para.strip():
                    chunks.append({
                        "id": f"{doc_id}-para-{i+1}",
                        "text": para,
                        "metadata": {
                            "title": f"{doc_id.replace('-', ' ').title()} (Para {i+1})",
                            "category": category,
                            "content_type": content_type,
                        }
                    })
    else:
        # Content fits in a single chunk
        chunks.append({
            "id": doc_id,
            "text": content,
            "metadata": {
                "title": doc_id.replace('-', ' ').title(),
                "category": category,
                "content_type": content_type
            }
        })
    
    logger.info(f"Created {len(chunks)} chunks from markdown content")
    return chunks

def process_text_content(content: str, filename: str, content_type: str = "information") -> List[Dict[str, Any]]:
    """Process text content into proper-sized chunks for embedding."""
    chunks = []
    doc_id = os.path.splitext(filename)[0].lower().replace(' ', '-')
    
    # Determine category without changing content_type
    category = "general"
    if "size" in filename.lower() or "sizing" in content.lower()[:500]:
        category = "sizing_guide"
    elif "maintenance" in filename.lower() or "care" in content.lower()[:500]:
        category = "maintenance"
    
    # Check if content exceeds token limit and needs chunking
    estimated_tokens = estimate_tokens(content)
    if estimated_tokens > 8000:
        # Split into smaller chunks using our utility function
        try:
            chunked_texts = chunk_text(content)
            
            # Create a document for each chunk
            for i, chunk_content in enumerate(chunked_texts):
                chunk_id = f"{doc_id}-chunk-{i+1}"
                
                # For first chunk, use original title; for others add part number
                title = doc_id.replace('-', ' ').title()
                if i > 0:
                    title += f" (Part {i+1})"
                    
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_content,
                    "metadata": {
                        "title": title,
                        "category": category,
                        "content_type": content_type,
                        "chunk_number": i+1,
                        "total_chunks": len(chunked_texts)
                    }
                })
        except Exception as e:
            logger.error(f"Error chunking content: {e}")
            # Fallback: Simple splitting by paragraphs
            paragraphs = content.split("\n\n")
            for i, para in enumerate(paragraphs[:10]):  # Limit to first 10 paragraphs if very large
                if para.strip():
                    chunks.append({
                        "id": f"{doc_id}-para-{i+1}",
                        "text": para,
                        "metadata": {
                            "title": f"{doc_id.replace('-', ' ').title()} (Para {i+1})",
                            "category": category,
                            "content_type": content_type,
                        }
                    })
    else:
        # Content fits in a single chunk
        chunks.append({
            "id": doc_id,
            "text": content,
            "metadata": {
                "title": doc_id.replace('-', ' ').title(),
                "category": category,
                "content_type": content_type,
            }
        })
        
    logger.info(f"Created {len(chunks)} chunks from text content")
    return chunks

def process_pdf_content(file_path: str, filename: str, content_type: str = "information") -> List[Dict[str, Any]]:
    """
    Process a PDF document into proper-sized chunks for embedding.
    
    Args:
        file_path (str): Path to the PDF file
        filename (str): Name of the file
        content_type (str): Type of content (information, product, brand)
        
    Returns:
        List[Dict[str, Any]]: List of processed chunks
    """
    chunks = []
    doc_id = os.path.splitext(filename)[0].lower().replace(' ', '-')
    
    # Determine category similar to other content types
    category = "general"
    if "size" in filename.lower():
        category = "sizing_guide"
    elif "maintenance" in filename.lower() or "care" in filename.lower():
        category = "maintenance"
    
    try:
        # Extract text from PDF
        pdf_text = ""
        page_texts = []
        
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            
            logger.info(f"Extracting text from PDF with {num_pages} pages")
            
            # Extract text from each page
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text.strip():
                    # Clean up common PDF extraction issues
                    page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
                    page_text = re.sub(r'(\w)-\s*\n(\w)', r'\1\2', page_text)  # Fix hyphenation
                    
                    # Store the page text
                    page_texts.append(page_text)
                    pdf_text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
        
        # Determine chunking strategy based on PDF size and structure
        logger.info(f"Total PDF text length: {len(pdf_text)} characters")
        
        # If PDF is small enough, keep it as a single document
        estimated_tokens = estimate_tokens(pdf_text)
        if estimated_tokens <= 2000:
            # PDF fits in a single chunk
            chunks.append({
                "id": doc_id,
                "text": pdf_text,
                "metadata": {
                    "title": doc_id.replace('-', ' ').title(),
                    "category": category,
                    "content_type": content_type,
                    "source": "pdf",
                    "pages": num_pages
                }
            })
        else:
            # Use two different chunking strategies
            
            # 1. Page-based chunking (good for clearly structured PDFs)
            if num_pages <= 20:  # Only use for shorter PDFs
                for page_num, page_text in enumerate(page_texts):
                    # Skip empty pages
                    if not page_text.strip():
                        continue
                        
                    # Create a chunk for this page
                    page_id = f"{doc_id}-page-{page_num + 1}"
                    chunks.append({
                        "id": page_id,
                        "text": f"--- Page {page_num + 1} ---\n\n{page_text}",
                        "metadata": {
                            "title": f"{doc_id.replace('-', ' ').title()} - Page {page_num + 1}",
                            "category": category,
                            "content_type": content_type,
                            "source": "pdf",
                            "page_number": page_num + 1,
                            "total_pages": num_pages
                        }
                    })
            
            # 2. Semantic chunking (better for content retrieval)
            # Use the existing chunk_text function that segments by sections/paragraphs
            semantic_chunks = chunk_text(pdf_text)
            
            for i, chunk_content in enumerate(semantic_chunks):
                chunk_id = f"{doc_id}-chunk-{i+1}"
                
                # For first chunk, use original title; for others add part number
                title = doc_id.replace('-', ' ').title()
                if i > 0:
                    title += f" (Part {i+1})"
                    
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_content,
                    "metadata": {
                        "title": title,
                        "category": category,
                        "content_type": content_type,
                        "source": "pdf",
                        "chunk_number": i+1,
                        "total_chunks": len(semantic_chunks)
                    }
                })
        
        logger.info(f"Created {len(chunks)} chunks from PDF document")
        
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {e}")
        import traceback
        traceback.print_exc()
    
    return chunks

def process_data_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a data file of any supported format into chunks.
    
    Args:
        file_path (str): Path to data file
        
    Returns:
        list: List of processed chunks
    """
    logger.info(f"Processing file: {file_path}")
    data_to_store = []
    
    # Determine content type from filename
    filename = os.path.basename(file_path).lower()
    content_type = "information"  # Default
    
    if "product" in filename:
        content_type = "product"
    elif "brand" in filename:
        content_type = "brand"

    # Add this in the process_data_file function after determining content_type

    # Log content type detection for clarity
    if "sizing" in filename or "size" in filename:
        logger.info(f"Detected sizing content in {filename} - assigning to information namespace")
        content_type = "information"
    elif "product" in filename:
        logger.info(f"Detected product content in {filename}")
        content_type = "product"
    elif "brand" in filename:
        logger.info(f"Detected brand content in {filename}")
        content_type = "brand"
    else:
        logger.info(f"No specific type detected for {filename} - defaulting to information")
        content_type = "information"
    
    # make sure the file is in the TARGET_NAMESPACES or TARGET_NAMESPACES is empty
    global TARGET_NAMESPACES
    should_process_file = not TARGET_NAMESPACES or TARGET_NAMESPACES.__len__() == 0
    if not should_process_file:
        if content_type == "product":
            should_process_file = "products" in TARGET_NAMESPACES
        elif content_type == "brand":
            should_process_file = "brand" in TARGET_NAMESPACES
        else:
            should_process_file = "information" in TARGET_NAMESPACES

    if not should_process_file:
        logger.info(f"Skipping file {filename} as it does not belong to the target namespaces")
        return data_to_store
    
    try:
        # Process JSON files
        if file_path.endswith(('.json')):
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    
                    # Check if it's product data (list of items with productUrl)
                    if isinstance(data, list) and all(isinstance(item, dict) and "productUrl" in item for item in data[:5]):
                        logger.info(f"Detected product catalog with {len(data)} items")
                        product_chunks = process_product_data(data)
                        data_to_store.extend(product_chunks)
                        
                    # Check if it's structured information data
                    elif isinstance(data, dict) and "sections" in data:
                        logger.info(f"Detected structured information with {len(data['sections'])} sections")
                        info_chunks = process_information_data(data, content_type)
                        data_to_store.extend(info_chunks)
                        
                    # Generic JSON processing
                    else:
                        logger.info("Processing as generic JSON data")
                        generic_chunks = process_generic_json(data, os.path.basename(file_path), content_type)
                        data_to_store.extend(generic_chunks)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON file {file_path}: {e}")
        
        # Process Markdown files
        elif file_path.endswith(('.md', '.markdown')):
            logger.info(f"Processing markdown file: {filename}")
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                md_chunks = process_markdown_content(content, filename, content_type)
                data_to_store.extend(md_chunks)
        
        # Process text files
        elif file_path.endswith(('.txt')):
            logger.info(f"Processing text file: {filename}")
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                # Process as text content
                md_chunks = process_text_content(content, filename, content_type)
                data_to_store.extend(md_chunks)
                
        # Process PDF files
        elif file_path.endswith(('.pdf')):
            logger.info(f"Processing PDF file: {filename}")
            pdf_chunks = process_pdf_content(file_path, filename, content_type)
            data_to_store.extend(pdf_chunks)
                
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        
    return data_to_store

# =========== PINECONE OPERATIONS ===========

def ensure_index_exists():
    """Create the Pinecone index if it doesn't exist."""
    try:
        print("📊 Checking if index exists... ", end="", flush=True)
        
        # Get list of indexes - compatible with different SDK versions
        try:
            # Try the current SDK approach
            indexes = pc.list_indexes()
            # Convert to list of names (different SDK versions return different formats)
            if isinstance(indexes, list):
                index_names = [idx["name"] for idx in indexes]
            else:
                # If it's an object with a names attribute
                index_names = indexes.names() if callable(getattr(indexes, "names", None)) else indexes.names
                
        except (TypeError, AttributeError):
            # Another approach for newer SDK
            indexes = pc.list_indexes()
            index_names = [index for index in indexes]
        
        if INDEX_NAME not in index_names:
            print("Creating new index...")
            try:
                # Try current SDK version approach
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=VECTOR_DIMENSION,
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "gcp",
                            "region": "us-central1",
                        }
                    }
                )
            except (TypeError, AttributeError) as e:
                logger.warning(f"First index creation attempt failed: {e}, trying alternate method")
                # Try older SDK version approach
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=VECTOR_DIMENSION,
                    metric="cosine", 
                    spec=PodSpec(
                        environment="gcp-starter"
                    )
                )
            
            # Wait for the index to be ready
            print("Waiting for index to be ready...")
            is_ready = False
            while not is_ready:
                try:
                    status = pc.describe_index(INDEX_NAME)
                    # Check status in a way that works with multiple SDK versions
                    if hasattr(status, "status"):
                        is_ready = status.status.get("ready", False)
                    else:
                        is_ready = status.get("ready", False)
                except Exception as e:
                    logger.warning(f"Error checking index readiness: {e}")
                
                if not is_ready:
                    time.sleep(1)
                    print(".", end="", flush=True)
                
            print("\nIndex created successfully ✓")
            logger.info(f"Index '{INDEX_NAME}' created successfully")
        else:
            print("Found existing index ✓")
            logger.info(f"Index '{INDEX_NAME}' already exists")
            
    except Exception as e:
        print("Failed ✗")
        logger.error(f"Failed to create or verify index: {e}")
        raise

def upload_data_to_pinecone(namespace_data: Dict[str, List[Dict[str, Any]]]):
    """
    Upload processed data to Pinecone using server-side vectorization.
    
    Args:
        namespace_data (dict): Dictionary mapping namespaces to data chunks
    """
    index = pc.Index(INDEX_NAME)
    
    # Get index stats to check which namespaces exist
    try:
        stats = index.describe_index_stats()
        existing_namespaces = list(stats.namespaces.keys()) if hasattr(stats, 'namespaces') and stats.namespaces else []
        logger.info(f"Found existing namespaces: {existing_namespaces}")
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        existing_namespaces = []
    
    # Clear each namespace before uploading (only if it exists)
    for namespace in namespace_data.keys():
        if namespace_data[namespace]:
            try:
                if namespace in existing_namespaces:
                    logger.info(f"Clearing existing namespace '{namespace}'...")
                    print(f"🗑️  Clearing '{namespace}' namespace... ", end="", flush=True)
                    index.delete(delete_all=True, namespace=namespace)
                    print("Done ✓")
                    logger.info(f"Namespace '{namespace}' cleared successfully")
                else:
                    logger.info(f"Namespace '{namespace}' doesn't exist yet, no need to clear")
                    print(f"🔍 Namespace '{namespace}' doesn't exist yet, will be created")
            except Exception as e:
                print("Failed ✗")
                logger.error(f"Failed to clear namespace '{namespace}': {e}")
    
    # Upload data for each namespace
    for namespace, data_to_store in namespace_data.items():
        if not data_to_store:
            logger.info(f"No data to upload for namespace '{namespace}'")
            continue
            
        total_chunks = len(data_to_store)
        logger.info(f"Uploading {total_chunks} document chunks to '{namespace}' namespace...")
        print(f"\n📤 Uploading to namespace: '{namespace}' ({total_chunks} chunks)")
        
        batch_size = BATCH_SIZE
        batches = [data_to_store[i:i + batch_size] for i in range(0, len(data_to_store), batch_size)]
        
        # Create progress bar for all batches
        with tqdm(total=len(batches), desc=f"Uploading {namespace}", unit="batch") as pbar:
            for batch_idx, batch in enumerate(batches):
                try:
                    batch_start_time = time.time()
                    
                    # log the first batch for debugging
                    if batch_idx == 0:
                        logger.info(f"First batch for namespace '{namespace}': {batch[0]}")
                    
                    # Create records with text in metadata for server-side embedding
                    records = []
                    for d in batch:
                        # Ensure text is a string (fix for PDF chunks that might return objects)
                        text_content = d["text"]
                        if isinstance(text_content, dict):
                            # If text is a dictionary (from chunk_text), extract the text field
                            logger.warning(f"Found dictionary instead of string for text in {d['id']} - converting to string")
                            if "text" in text_content:
                                text_content = text_content["text"]
                            else:
                                text_content = json.dumps(text_content)
                        elif not isinstance(text_content, str):
                            # For any other non-string type, convert to string
                            logger.warning(f"Converting non-string text ({type(text_content)}) to string for {d['id']}")
                            text_content = str(text_content)
                        
                        records.append({
                            "_id": d["id"],
                            "text": text_content,  # Use sanitized text content
                            "category": d["metadata"].get("categories", "General"),
                            "metadata": json.dumps(d["metadata"])
                        })
                        
                        # if alternate_text exists, also add a record
                        if "alternate_text" in d:
                            alt_text = d["alternate_text"]
                            # Also ensure alternate_text is a string
                            if not isinstance(alt_text, str):
                                alt_text = str(alt_text)
                                
                            records.append({
                                "_id": d["id"] + "-alt",
                                "text": alt_text,
                                "category": d["metadata"].get("categories", "General"),
                                "metadata": json.dumps(d["metadata"])
                            })
                    
                    # Upload to Pinecone using source_field for server-side embedding
                    index.upsert_records(
                        namespace=namespace,
                        records=records, 
                    )
                    
                    batch_time = time.time() - batch_start_time
                    
                    # Update progress description back to normal with stats
                    chunks_done = min((batch_idx + 1) * batch_size, total_chunks)
                    pbar.set_description(
                        f"Uploading {namespace}: {chunks_done}/{total_chunks} chunks, {batch_time:.1f}s/batch"
                    )
                    
                    # Update progress
                    pbar.update(1)
                    
                    # Sleep between batches to avoid rate limits, but only if more batches left
                    if batch_idx < len(batches) - 1:
                        time.sleep(BATCH_SLEEP_TIME)
                        
                except Exception as e:
                    logger.error(f"Error uploading batch {batch_idx+1}/{len(batches)}: {e}")
                    import traceback
                    traceback.print_exc()
                    if batch_idx < len(batches) - 1:
                        time.sleep(BATCH_SLEEP_TIME * 2)

def generate_sparse_vector(text: str):
    """
    Generate a simple sparse vector using token frequencies.
    In production, use better methods like BM25 or SPLADE.
    """
    # Tokenize the text (simple lowercase word tokenization)
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    # Count token frequencies
    token_counts = {}
    for token in tokens:
        if len(token) > 2:  # Skip very short tokens
            token_counts[token] = token_counts.get(token, 0) + 1
    
    # Create sparse indices and values
    indices = []
    values = []
    
    # Hash tokens to indices (simple approach)
    vocabulary_size = 100000  # Large sparse dimension size
    for token, count in token_counts.items():
        # Hash the token to an index
        index = hash(token) % vocabulary_size
        indices.append(index)
        values.append(float(count))  # Use count as weight
    
    return {"indices": indices, "values": values}

def generate_embeddings_with_pinecone(texts):
    """Generate embeddings using Pinecone's integrated embedding service"""
    try:
        # Safety checks for token limits (existing code)
        filtered_texts = []
        skipped = 0
        
        for i, text in enumerate(texts):
            est_tokens = estimate_tokens(text)
            
            if est_tokens > 2000:  # llama-text-embed-v2 limit is 2048
                logger.warning(f"Text at index {i} exceeds token limit ({est_tokens} tokens). Truncating.")
                text = text[:7600]  # ~1900 tokens (4 chars per token)
                filtered_texts.append(text)
                skipped += 1
            else:
                filtered_texts.append(text)
        
        if skipped > 0:
            logger.warning(f"Truncated {skipped} texts that exceeded token limits")
            
        # Process in batches of 96 (Pinecone's limit)
        batch_size = 96
        all_embeddings = []
        
        # Process each batch
        for i in range(0, len(filtered_texts), batch_size):
            batch_texts = filtered_texts[i:i+batch_size]
            
            print(f"Sending batch {i//batch_size + 1} with {len(batch_texts)} texts to Pinecone embedding API")
            
            # Use the base model name with dimension in parameters
            response = pc.inference.embed(
                model="llama-text-embed-v2",  # Base model name only
                inputs=batch_texts,
                parameters={
                    "input_type": "passage", 
                    "truncate": "END",
                    "dimension": 2048  # Specify dimension in parameters
                }
            )
            
            print(f"Response type: {type(response)}")
            
            # Access the data field as per documentation
            if hasattr(response, 'data') and response.data:
                print(f"Found {len(response.data)} embeddings in response.data")
                
                # Extract just the values array from each embedding
                for embedding in response.data:
                    if hasattr(embedding, 'values'):
                        all_embeddings.append(embedding.values)
                    elif isinstance(embedding, dict) and 'values' in embedding:
                        all_embeddings.append(embedding['values'])
                    else:
                        print(f"Unexpected embedding structure: {type(embedding)}")
                        raise ValueError(f"Could not extract values from embedding: {embedding}")
            else:
                print("No 'data' attribute in response or it's empty")
                raise ValueError("No embeddings found in response")
            
        return all_embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings with Pinecone: {e}")
        import traceback
        traceback.print_exc()
        raise

# class PineconeRAG:
#     """
#     Simple RAG implementation for testing Pinecone searches.
#     """
    
#     def __init__(self, index_name, model_name=None, debug=False):
#         """Initialize with index name and optional model name for embeddings"""
#         self.index_name = index_name
#         self.model_name = model_name or EMBEDDING_MODEL
#         self.index = pc.Index(index_name)
#         self.debug = debug
    
#     def search(self, query, top_k=25, top_n=5, min_score=0.5, namespace=""):
#         """Search pinecone index with the given query"""
        
#         # Search the index
#         # results = self.index.query(
#         #     namespace=namespace,
#         #     vector=embedding,
#         #     top_k=top_k,
#         #     include_values=False,
#         #     include_metadata=True
#         # )
        
#         indices = [
#             self.index,
#             pc.Index("specialized-detailed")
#         ]
        
#         # Format the results
#         formatted_results = []

#         # search each index
#         for i, index in enumerate(indices):
#             index_name = 'specialized-llama-1024'
#             if i == 1:
#                 index_name = 'specialized-detailed'
#             print(f"Searching index: {index_name}")
#             print(f"Searching namespace: {namespace}")
#             print(query)
        
#             start_time = time.time()
#             results = index.search_records(
#                 namespace=namespace, 
#                 query=SearchQuery(
#                     inputs={
#                         "text": query
#                     }, 
#                     top_k=top_k
#                 ),
#                 rerank=SearchRerank(
#                     model=RerankModel.Pinecone_Rerank_V0, # Bge_Reranker_V2_M3
#                     rank_fields=["text"],
#                     top_n=top_n,
#                     parameters={
#                         "truncate": "END"
#                     }
#                 ),
#                 # rerank=SearchRerank(
#                 #     model=RerankModel.Cohere_Rerank_3_5, #  Bge_Reranker_V2_M3, # : "bge-reranker-v2-m3",
#                 #     rank_fields=["text"],
#                 #     top_n=top_n
#                 # ),
#                 fields=["category", "metadata"],
#             )
#             end_time = time.time()
#             print(f"Search time: {end_time - start_time:.2f}s ({index_name})")
#             start_time2 = time.time()
#             results2 = index.search_records(
#                 namespace=namespace, 
#                 query=SearchQuery(
#                     inputs={
#                         "text": query
#                     }, 
#                     top_k=top_k
#                 ),
#                 rerank=SearchRerank(
#                     model=RerankModel.Cohere_Rerank_3_5, #  Bge_Reranker_V2_M3, # : "bge-reranker-v2-m3",
#                     rank_fields=["text"],
#                     top_n=top_n
#                 ),
#                 fields=["category", "metadata"],
#             )
#             end_time2 = time.time()
#             print(f"Search time2: {end_time2 - start_time2:.2f}s ({index_name})")
            
#             start_time3 = time.time()
#             results3 = index.search_records(
#                 namespace=namespace, 
#                 query=SearchQuery(
#                     inputs={
#                         "text": query
#                     }, 
#                     top_k=top_k
#                 ),
#                 fields=["category", "metadata"],
#             )
#             end_time3 = time.time()
#             print(f"Search time3: {end_time3 - start_time3:.2f}s ({index_name})")
#             # print(f"Search results: {results}")
                        
#             if not results.result.hits:
#                 print("No results found")
#                 return formatted_results
            
#             # print out the top_k results from each set with the: _id, _score, metadata.get("name")
#             print(f"Top {top_k} results from {index_name}:")
#             for i, match in enumerate(results.result.hits[:top_n]):
#                 metadata = json.loads(match.fields.get('metadata'))
#                 print(f"  {i+1}. ID: {match._id}, Score: {match._score:.4f}, Name: {metadata.get('name')}")
#             print("")

#             for i, match in enumerate(results2.result.hits[:top_n]):
#                 metadata = json.loads(match.fields.get('metadata'))
#                 print(f"  {i+1}. ID: {match._id}, Score: {match._score:.4f}, Name: {metadata.get('name')}")
#             print("")
            
#             for i, match in enumerate(results3.result.hits[:top_n]):
#                 metadata = json.loads(match.fields.get('metadata'))
#                 print(f"  {i+1}. ID: {match._id}, Score: {match._score:.4f}, Name: {metadata.get('name')}")
#             print("")

#             # # compare the two results scores
#             # for i, match in enumerate(results.result.hits):
#             #     metadata1 = json.loads(match.fields.get('metadata'))
#             #     metadata2 = json.loads(results2.result.hits[i].fields.get('metadata'))
#             #     metadata3 = json.loads(results3.result.hits[i].fields.get('metadata'))
#             #     if metadata1.get('name') != metadata2.get('name'):
#             #         print(f"Name mismatch: {metadata1.get('name')} ({match._score}) vs {metadata2.get('name')} ({results2.result.hits[i]._score}) vs {metadata3.get('name')} ({results3.result.hits[i]._score})")
            
#             # In debug mode, print all results with their scores
#             if self.debug:
#                 print(f"\n  [DEBUG] Found {len(results.result.hits)} total matches:")
#                 for i, match in enumerate(results.result.hits):
#                     print(f"  [DEBUG] Match {i+1}: score={match._score:.4f}, id={match._id}")
#                     metadata_str = match.fields.get('metadata')
#                     if metadata_str:
#                         metadata = json.loads(metadata_str)
#                         if namespace == "products":
#                             print(f"  [DEBUG]   Metadata: {metadata.get('name')}")
#                         else:
#                             print(f"  [DEBUG]   Metadata: {metadata.get('title')}")
            
#             # Filter results by score threshold
#             for match in results.result.hits:
#                 if match._score >= min_score:
#                     metadata_str = match.fields.get('metadata')
#                     if metadata_str:
#                         metadata = json.loads(metadata_str)
#                     else:
#                         metadata = {}
#                     formatted_results.append({
#                         'id': match._id,
#                         'score': match._score,
#                         'metadata': metadata
#                     })
            
#         return formatted_results
    

async def test_queries(custom_query=None, min_score=0.35, min_n=3):
    """Run test queries to verify the index is working correctly."""
    # Use custom query if provided, otherwise use default queries
    if custom_query:
        queries = [custom_query]
    else:
        queries = [
            "What mountain bikes do you have under $3000?",
            "Tell me about your best road helmets",
            "Do you have carbon fiber gravel bikes?",
            "I need shoes for mountain biking",
            "What's the difference between the Tarmac and Roubaix?",
            "Show me bikes in size medium",
            "How do I maintain my bike chain?",
            "Tell me about the Specialized company history"
        ]
    
    top_k = 15
    top_n = 5
    min_score = min_score  # Use provided min_score
    
    # # Initialize the conversation analyzer
    # from spence.conversation.analyzer import ConversationAnalyzer
    # from livekit.agents.llm import ChatMessage
    # from livekit.plugins import google, openai
    # import google.generativeai as genai

    # model_name="gemini-2.0-flash"
    # is_google_model = model_name.startswith("gemini")
    
    # # Set up mock model for analyzer
    # if not os.environ.get("GOOGLE_API_KEY"):
    #     print("Warning: GOOGLE_API_KEY not set, using mock LLM")
    #     llm_model = llm.MockLLM()
    # else:
    #     genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    #     # Configure LLM
    #     if is_google_model:    
    #         llm_model = google.LLM(model=model_name)
    #     else:
    #         llm_model = openai.LLM(model="gpt-4o-mini")
    
    # # Initialize analyzer
    # analyzer = ConversationAnalyzer(
    #     model=None,  # No need for voice pipeline in test
    #     llm_model=llm_model,
    #     user_id="test-user"
    # )
    analyzer = None
    
    # Use the PineconeRAG class with debug mode enabled
    rag = PineconeRAG(index_name=INDEX_NAME, model_name=EMBEDDING_MODEL, debug=True)
    
    # Test each query
    print("\n========== TESTING RAG SYSTEM WITH CONVERSATION ANALYSIS ==========")
    for query in queries:
        print(f"\n\nQUERY: {query}")
        
        if analyzer:
            # Simulate a conversation with this query
            await simulate_conversation_with_analyzer(analyzer, query)
            # await simulate_conversation(query)
        
        # # Get filters from analyzer
        if analyzer:
            filters = await analyzer.get_filters()
        else:
            filters = {}
        
        # # Get search context if available
        search_context = filters.get("search_context", "")
        # search_context = None
        
        # Create enhanced query with search context
        enhanced_query = query
        if (search_context):
            print(f"\nEnhanced with context: '{search_context}'")
            enhanced_query = f"{query} {search_context}"
        
        # Get product categories and other filters for search
        product_categories = filters.get("product_categories", [])
        bike_type = filters.get("bike_type", [])
        price_range = filters.get("price_range", "")
        
        print(f"\nExtracted filters:\n- Categories: {', '.join(product_categories) if product_categories else 'None'}")
        print(f"- Bike types: {', '.join(bike_type) if bike_type else 'None'}")
        print(f"- Price range: {price_range or 'Not specified'}")
        
        namespaces = NAMESPACES
        if TARGET_NAMESPACES and len(TARGET_NAMESPACES) > 0:
            namespaces = TARGET_NAMESPACES
        
        print(f"\n----- RAG Results using enhanced query -----")
        try:
            # Use the enhanced query for search
            search_ranked_results = await rag.search_ranked(
                enhanced_query,
                top_k=top_k,
                top_n=top_n,
                min_score=min_score,
                min_n=min_n,
                namespaces=namespaces
            )
            print(f"\n  Found {len(search_ranked_results)} total results across all namespaces")
            
            # print out the results with the: _id, _score, metadata.get("name")
            for i, result in enumerate(search_ranked_results):
                print(f"  {i+1}. ID: {result['id']}, Score: {result['score']:.4f}")
                metadata = result.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                if metadata.get('content_type') == 'product':
                    print(f"     Product: {metadata.get('name', 'Unknown')}")
                    categories = metadata.get('categories', [])
                    if categories:
                        print(f"     Categories: {', '.join(categories)}")
                    print(f"     Price: {metadata.get('original_price', 'Unknown')}")
                else:
                    print(f"     Title: {metadata.get('title', 'Unknown')}")
        except Exception as e:
            print(f"  Error testing query: {e}")
            import traceback
            traceback.print_exc()
        
        # Reset analyzer for the next query
        if analyzer:
            await analyzer.reset_filters()
    
    print("\n========================================")

async def simulate_conversation(query):
    from livekit.agents.llm import ChatMessage
    from spence.conversation.analyzer import ConversationAnalyzer
    import asyncio
    
    """Simulate a brief conversation to build context in the analyzer"""
    # Create a simple conversation with context for the query
    sample_conversation = []
    analyzer = ConversationAnalyzer
    
    # Add system message
    sample_conversation.append({
        "role": "system",
        "content": "You are Spence, a knowledgeable sales assistant for Specialized bikes."
    })
    
    # Add some context based on the query
    if "mountain" in query.lower():
        sample_conversation.extend([
            ChatMessage(role="user", content="Hi, I'm looking for a mountain bike."),
            ChatMessage(role="assistant", content="Great! I can help you find the perfect mountain bike. What kind of terrain will you be riding on?"),
            ChatMessage(role="user", content="Mostly trails and some downhill riding.")
        ])
    elif "road" in query.lower():
        sample_conversation.extend([
            ChatMessage(role="user", content="I need a good road bike for long rides."),
            ChatMessage(role="assistant", content="I'd be happy to help you find a road bike. Are you looking for something for racing or more for endurance rides?"),
            ChatMessage(role="user", content="More for longer endurance rides.")
        ])
    elif "helmet" in query.lower():
        sample_conversation.extend([
            ChatMessage(role="user", content="I need to protect my head while riding."),
            ChatMessage(role="assistant", content="Safety is definitely important! What kind of riding do you do most often?"),
            ChatMessage(role="user", content="I do road riding mostly.")
        ])
    else:
        # Generic conversation starter
        sample_conversation.extend([
            {"role":"user", "content":"Hi, I'm looking for some cycling gear."},
            {"role":"assistant", "content":"Welcome! I'd be happy to help you find what you need. What kind of riding do you do?"},
            {"role":"user", "content":"I ride a mix of road and trails."}
        ])
    
    # Feed conversation into analyzer
    analyzer._conversation = sample_conversation
    # for message in sample_conversation:
    #     if message.role == "user":
    #         analyzer._log_q.put_nowait({"event": "user_speech_committed", "message": message})
    #     elif message.role == "assistant":
    #         analyzer._log_q.put_nowait({"event": "agent_speech_committed", "message": message})
    
    # # Add a slight delay for processing
    # await asyncio.sleep(0.5)
    
    # # Add the actual query as the final user message
    # query_message = ChatMessage(role="user", content=query)
    # analyzer._log_q.put_nowait({"event": "user_speech_committed", "message": query_message})
    
    # Run analysis explicitly
    await analyzer._run_analysis()
    
    # # Generate search context
    # await analyzer._generate_search_context()
    
    print(f"Conversation analysis complete: {analyzer.conversation_filters}")

async def simulate_conversation_with_analyzer(analyzer, query):
    from livekit.agents.llm import ChatMessage
    import asyncio
    
    """Simulate a brief conversation to build context in the analyzer"""
    # Create a simple conversation with context for the query
    sample_conversation = []
    
    # Add system message
    sample_conversation.append(ChatMessage(
        role="system",
        content="You are Spence, a knowledgeable sales assistant for Specialized bikes."
    ))
    
    # Add some context based on the query
    if "mountain" in query.lower():
        sample_conversation.extend([
            ChatMessage(role="user", content="Hi, I'm looking for a mountain bike."),
            ChatMessage(role="assistant", content="Great! I can help you find the perfect mountain bike. What kind of terrain will you be riding on?"),
            ChatMessage(role="user", content="Mostly trails and some downhill riding.")
        ])
    elif "road" in query.lower():
        sample_conversation.extend([
            ChatMessage(role="user", content="I need a good road bike for long rides."),
            ChatMessage(role="assistant", content="I'd be happy to help you find a road bike. Are you looking for something for racing or more for endurance rides?"),
            ChatMessage(role="user", content="More for longer endurance rides.")
        ])
    elif "helmet" in query.lower():
        sample_conversation.extend([
            ChatMessage(role="user", content="I need to protect my head while riding."),
            ChatMessage(role="assistant", content="Safety is definitely important! What kind of riding do you do most often?"),
            ChatMessage(role="user", content="I do road riding mostly.")
        ])
    else:
        # Generic conversation starter
        sample_conversation.extend([
            ChatMessage(role="user", content="Hi, I'm looking for some cycling gear."),
            ChatMessage(role="assistant", content="Welcome! I'd be happy to help you find what you need. What kind of riding do you do?"),
            ChatMessage(role="user", content="I ride a mix of road and trails.")
        ])
    
    # Feed conversation into analyzer
    analyzer._conversation = sample_conversation
    # for message in sample_conversation:
    #     if message.role == "user":
    #         analyzer._log_q.put_nowait({"event": "user_speech_committed", "message": message})
    #     elif message.role == "assistant":
    #         analyzer._log_q.put_nowait({"event": "agent_speech_committed", "message": message})
    
    # # Add a slight delay for processing
    # await asyncio.sleep(0.5)
    
    # # Add the actual query as the final user message
    # query_message = ChatMessage(role="user", content=query)
    # analyzer._log_q.put_nowait({"event": "user_speech_committed", "message": query_message})
    
    # Run analysis explicitly
    await analyzer._run_analysis()
    
    # # Generate search context
    # await analyzer._generate_search_context()
    
    print(f"Conversation analysis complete: {analyzer.conversation_filters}")

# Update the parse_args function to include a RAG test flag

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Pinecone Vector Database Setup for Multi-Namespace RAG System"
    )
    
    # Create a group of mutually exclusive action arguments
    action_group = parser.add_argument_group("actions (at least one required)")
    
    # Add import/upload flag - explicit opt-in for data import
    action_group.add_argument(
        "--import",
        action="store_true",
        dest="import_data",
        help="Import and upload data to Pinecone (explicit opt-in required)"
    )
    
    # Add test flag 
    action_group.add_argument(
        "--test",
        action="store_true",
        help="Run test queries against the database"
    )
    
    # Add RAG test flag (another action option)
    action_group.add_argument(
        "--rag-test",
        action="store_true",
        help="Run a quick test of the RAG system functionality"
    )
    
    # Additional options
    parser.add_argument(
        "--namespaces",
        nargs="+",
        choices=NAMESPACES,
        help="Specify which namespaces to process (default: all)"
    )
    
    # Add custom query flag
    parser.add_argument(
        "--query",
        type=str,
        help="Run a single custom query instead of the default test queries"
    )
    
    # Add min-score flag
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.25,
        help="Minimum similarity score threshold (default: 0.25)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate that at least one action was specified
    if not (args.import_data or args.test or args.rag_test):
        parser.print_help()
        print("\n⚠️  Error: You must specify at least one action (--import, --test, or --rag-test)")
        sys.exit(1)
        
    return args

# Add this function before the main function

async def test_rag_functionality():
    """Run a simple test of the RAG system functionality"""
    print("\n===== TESTING RAG FUNCTIONALITY =====")
    try:
        print("Initializing RAG system...")
        rag = PineconeRAG(
            index_name=INDEX_NAME, 
            model_name=EMBEDDING_MODEL,
            namespace="specialized"
        )
        
        test_queries = [
            "Tell me about mountain bikes",
            "What are the best road helmets?",
            "How do I maintain my bike chain?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            
            # Search across all namespaces
            results = await rag.multi_namespace_search(
                query=query,
                namespaces=NAMESPACES,
                top_k=3,
                min_score=0.3
            )
            
            # Print results summary
            for namespace, docs in results.items():
                print(f"\n  Namespace '{namespace}': {len(docs)} results")
                for i, doc in enumerate(docs[:1]):  # Show top result only
                    print(f"    {i+1}. Score: {doc.get('score', 0):.3f}")
                    if doc.get('metadata', {}).get('content_type') == 'product':
                        print(f"       Product: {doc.get('metadata', {}).get('name', 'Unknown')}")
                    else:
                        print(f"       Title: {doc.get('metadata', {}).get('title', 'Unknown')}")
        
        print("\n✅ RAG functionality test completed successfully")
        return True
    except Exception as e:
        print(f"\n❌ RAG functionality test failed: {e}")
        logger.error(f"RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# =========== MAIN EXECUTION ===========

def main():
    """Main execution function."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Fix variable handling in main()

        # Set global variables based on arguments
        global IMPORT_DATA, TARGET_NAMESPACES
        IMPORT_DATA = args.import_data

        if args.namespaces:
            # Only override TARGET_NAMESPACES with namespaces that exist in NAMESPACES
            TARGET_NAMESPACES = [ns for ns in args.namespaces if ns in NAMESPACES]
            print(f"🎯 Targeting namespaces: {', '.join(TARGET_NAMESPACES)}")
        else:
            # Default to all namespaces
            TARGET_NAMESPACES = NAMESPACES.copy()
        
        # Display banner for better visibility
        print("\n" + "="*60)
        print(" PINECONE RAG SETUP - MULTI-NAMESPACE INGESTION ".center(60, "="))
        print("="*60 + "\n")
        
        print(f"🏗️  Initializing setup for index: {INDEX_NAME}")
        
        # Ensure index exists regardless of mode
        print("📊 Checking if index exists... ", end="", flush=True)
        ensure_index_exists()
        print("Done ✓")
        
        # Import and upload data if specified
        if IMPORT_DATA:
            # Create storage for different namespaces (only targeted ones)
            namespace_data = {namespace: [] for namespace in NAMESPACES if namespace in TARGET_NAMESPACES}

            # Process files with progress bar
            file_count = 0
            total_chunks = 0

            print(f"\n📂 Data directory: {DATA_DIRECTORY}")
            print(f"🔍 Scanning for files...")
            data_files = []
            for filename in os.listdir(DATA_DIRECTORY):
                file_path = os.path.join(DATA_DIRECTORY, filename)
                if os.path.isfile(file_path) and (filename.endswith('.json') or filename.endswith('.jsonl') or filename.endswith('.txt') or filename.endswith('.csv') or filename.endswith('.tsv') or filename.endswith('.md') or filename.endswith('.pdf')):
                    data_files.append(file_path)

            print(f"📄 Found {len(data_files)} data files to process")

            with tqdm(total=len(data_files), desc="Processing files", unit="file") as pbar:
                for file_path in data_files:
                    filename = os.path.basename(file_path)
                    pbar.set_description(f"Processing {filename}")
                    
                    try:
                        # Process the file
                        file_chunks = process_data_file(file_path)
                        file_count += 1
                        
                        # Assign chunks to correct namespace (only if it's a targeted namespace)
                        for chunk in file_chunks:
                            content_type = chunk.get("metadata", {}).get("content_type", "information")
                            
                            # Determine the appropriate namespace
                            if content_type == "product":
                                namespace = "products"
                            elif content_type == "brand":
                                namespace = "brand"
                            else:
                                namespace = "information"
                            
                            # Only add chunks to targeted namespaces
                            if namespace in TARGET_NAMESPACES:
                                namespace_data[namespace].append(chunk)
                                total_chunks += 1
                            
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        pbar.update(1)
            
            # Show summary of processed data
            print("\n📊 Processing Summary:")
            for namespace, chunks in namespace_data.items():
                chunk_count = len(chunks)
                print(f"   - {namespace}: {chunk_count} chunks")
            
            print(f"\n💾 Total: {total_chunks} chunks from {file_count} files")
            
            # Upload to Pinecone
            print("\n🚀 Uploading data to Pinecone...")
            upload_data_to_pinecone(namespace_data)
            
            # Get index statistics
            print("\n📊 Getting index statistics... ", end="", flush=True)
            try:
                index = pc.Index(INDEX_NAME)
                stats = index.describe_index_stats()
                print("Done ✓")
                
                # Show vector counts by namespace
                print("\n🔢 Index Statistics:")
                print(f"   - Total vectors: {stats.total_vector_count}")
                
                if hasattr(stats, 'namespaces') and stats.namespaces:
                    for ns, ns_stats in stats.namespaces.items():
                        print(f"   - {ns}: {ns_stats.vector_count} vectors")
                else:
                    print("   - No namespace statistics available")
                    
            except Exception as e:
                print("Failed ✗")
                logger.error(f"Failed to get index statistics: {e}")
        
        # Run test queries
        if args.test:
            import asyncio
            print("\n🧪 Running test queries to verify system...")
            if args.query:
                asyncio.run(test_queries(custom_query=args.query, min_score=args.min_score))
            else:
                asyncio.run(test_queries(min_score=args.min_score))
        
        if args.rag_test:
            print("\n🧪 Testing RAG functionality...")
            import asyncio
            asyncio.run(test_rag_functionality())
        
        print("\n✅ Setup completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)