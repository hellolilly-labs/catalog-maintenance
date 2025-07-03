"""
Utilities for converting Python objects to Markdown format.
"""

import json
import re
from typing import Any, Dict, List, Set, Union

def _key_to_friendly_name(key: str) -> str:
    """
    Convert a programmatic key to a human-friendly name.
    
    Examples:
        user_id -> User ID
        productName -> Product Name
        item_price_usd -> Item Price USD
        created_at -> Created At
    """
    # Handle special abbreviations that should be uppercase
    common_abbr = {
        "id": "ID", 
        "url": "URL",
        "uri": "URI",
        "api": "API",
        "ui": "UI",
        "ux": "UX",
        "cdn": "CDN",
        "ip": "IP",
        "seo": "SEO",
        "http": "HTTP",
        "https": "HTTPS",
        "html": "HTML",
        "css": "CSS",
        "json": "JSON",
        "xml": "XML",
        "sql": "SQL",
        "usd": "USD",
        "eur": "EUR",
        "gbp": "GBP",
        "pdf": "PDF"
    }
    
    # Replace underscores and hyphens with spaces
    name = re.sub(r'[_\-]', ' ', key)
    
    # Split by camelCase (insert space before capitals)
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    
    # Title case the result (capitalize first letter of each word)
    words = name.split()
    result = []
    
    for word in words:
        # Check if the word is a common abbreviation
        word_lower = word.lower()
        if word_lower in common_abbr:
            result.append(common_abbr[word_lower])
        else:
            result.append(word.capitalize())
    
    return " ".join(result)

def obj_to_markdown(obj: Any, title: str = None, level: int = 1, convert_keys: bool = True) -> str:
    """
    Convert a Python object to a readable Markdown representation.
    
    Args:
        obj: The object to convert
        title: Optional title for the markdown section
        level: Heading level for nested objects
        convert_keys: If True, convert keys to friendly names
    
    Returns:
        str: Markdown formatted string representation
    """
    md_lines = []
    
    # Add title if provided
    if title and level == 1:
        friendly_title = _key_to_friendly_name(title) if convert_keys else title
        md_lines.append(f"# {friendly_title}")
        md_lines.append("")
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            friendly_key = _key_to_friendly_name(key) if convert_keys else key
            
            # Handle nested dictionaries with subheadings
            if isinstance(value, dict) and value:
                md_lines.append(f"{'#' * (level + 1)} {friendly_key}")
                md_lines.append("")
                md_lines.append(obj_to_markdown(value, level=level+1, convert_keys=convert_keys))
            
            # Handle lists specially
            elif isinstance(value, (list, set)) and value:
                md_lines.append(f"**{friendly_key}**:")
                md_lines.append("")
                
                for item in value:
                    if isinstance(item, dict):
                        md_lines.append(obj_to_markdown(item, level=level+1, convert_keys=convert_keys))
                        md_lines.append("---")
                    else:
                        md_lines.append(f"- {item}")
                md_lines.append("")
            
            # Simple key/value pairs
            else:
                # Skip empty values
                if value is None or (isinstance(value, (str, list, dict)) and not value):
                    continue
                    
                if isinstance(value, set):
                    value = list(value)
                    
                # Format the value based on type
                if isinstance(value, (int, float, bool, str)):
                    md_lines.append(f"**{friendly_key}**: {value}")
                else:
                    md_lines.append(f"**{friendly_key}**: {value}")
    
    elif isinstance(obj, (list, set)):
        for item in obj:
            if isinstance(item, dict):
                md_lines.append(obj_to_markdown(item, level=level+1, convert_keys=convert_keys))
                md_lines.append("---")
            else:
                md_lines.append(f"- {item}")
    else:
        md_lines.append(str(obj))
    
    return "\n".join(md_lines)


def url_context_to_markdown(url_data: Dict[str, Any]) -> str:
    """
    Specialized function to format URL context data as readable markdown.
    
    Args:
        url_data: Dictionary containing URL context information
        
    Returns:
        str: Markdown formatted string with URL context
    """
    if not url_data:
        return ""
        
    # Use the generic function but with specialized formatting for URL data
    return obj_to_markdown(url_data, title="Current Page Information", convert_keys=True)
