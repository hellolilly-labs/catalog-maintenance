"""
Test suite for descriptor generation system

Tests for LLM-powered descriptor and sizing generation including:
- LLM-based vertical detection capabilities
- LLMFactory integration
- Error handling and edge cases
- JSON response validation
"""

import json
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from src.descriptor import (
    DescriptorGenerator, 
    get_descriptor_generator,
    generate_product_descriptor,
    generate_product_sizing
)
from src.models.product import Product
from src.llm.errors import LLMError, TokenLimitError, ModelNotFoundError


class TestDescriptorGenerator:
    """Main test class for DescriptorGenerator functionality"""
    
    @pytest.fixture
    def generator(self):
        """Create a test descriptor generator with mocked LLMFactory"""
        generator = DescriptorGenerator()
        return generator
    
    @pytest.fixture
    def sample_cycling_product(self):
        """Create a sample cycling product for testing"""
        return Product(
            id="test-bike-123",
            name="Tarmac SL7 Expert",
            brand="specialized.com",
            categories=["Road Bikes", "Performance"],
            originalPrice="$4,000",
            colors=["Gloss Red", "Satin Black"],
            sizes=["52cm", "54cm", "56cm", "58cm"],
            highlights=["Carbon Frame", "Shimano 105", "Tubeless Ready"],
            description="High-performance road bike with carbon frame",
            specifications={
                "Frame": {"material": "Carbon", "geometry": "Endurance"},
                "Drivetrain": {"brand": "Shimano", "model": "105"}
            }
        )
    
    @pytest.fixture
    def sample_fashion_product(self):
        """Create a sample fashion product for testing"""
        return Product(
            id="test-shirt-456",
            name="Premium Cotton T-Shirt",
            brand="fashion-brand.com",
            categories=["Clothing", "Casual Wear"],
            originalPrice="$45",
            colors=["Black", "White", "Navy"],
            sizes=["S", "M", "L", "XL"],
            highlights=["100% Cotton", "Pre-shrunk", "Classic Fit"],
            description="Comfortable everyday t-shirt in premium cotton",
            specifications={"Material": {"composition": "100% Cotton", "weight": "180gsm"}}
        )


class TestVerticalDetection:
    """Test LLM-based vertical detection functionality"""
    
    @pytest.mark.asyncio
    async def test_detect_brand_vertical_cycling(self):
        """Test LLM-based detection of cycling brand"""
        with patch('src.llm.simple_factory.LLMFactory.chat_completion') as mock_chat:
            mock_chat.return_value = {
                "content": "cycling",
                "usage": {"total_tokens": 20}
            }
            
            generator = DescriptorGenerator()
            
            cycling_product = Product(
                id="bike-1",
                name="Mountain Bike Pro",
                brand="specialized.com",
                categories=["Cycling", "Mountain Bikes"],
                highlights=["Carbon frame", "Shimano gears"]
            )
            
            vertical = await generator.detect_brand_vertical(cycling_product)
            assert vertical == "cycling"
            
            # Verify LLM was called for brand research
            mock_chat.assert_called_once()
            call_args = mock_chat.call_args[1]
            assert call_args["task"] == "brand_research"
            assert call_args["temperature"] == 0.1
    
    @pytest.mark.asyncio
    async def test_detect_brand_vertical_caching(self):
        """Test that brand vertical detection uses caching"""
        with patch('src.llm.simple_factory.LLMFactory.chat_completion') as mock_chat:
            mock_chat.return_value = {
                "content": "fashion", 
                "usage": {"total_tokens": 15}
            }
            
            generator = DescriptorGenerator()
            
            product = Product(
                id="shirt-1",
                name="Designer Shirt",
                brand="fashion-brand.com",
                categories=["Clothing"]
            )
            
            # First call should hit LLM
            vertical1 = await generator.detect_brand_vertical(product)
            assert vertical1 == "fashion"
            assert mock_chat.call_count == 1
            
            # Second call should use cache
            vertical2 = await generator.detect_brand_vertical(product) 
            assert vertical2 == "fashion"
            assert mock_chat.call_count == 1  # No additional LLM call
    
    @pytest.mark.asyncio
    async def test_detect_product_subvertical(self):
        """Test LLM-based product sub-vertical detection"""
        with patch('src.llm.simple_factory.LLMFactory.chat_completion') as mock_chat:
            mock_chat.return_value = {
                "content": "road bikes",
                "usage": {"total_tokens": 25}
            }
            
            generator = DescriptorGenerator()
            
            product = Product(
                id="bike-1",
                name="Road Racing Bike",
                categories=["Road Bikes"],
                highlights=["Aerodynamic", "Racing geometry"]
            )
            
            subvertical = await generator.detect_product_subvertical(product, "cycling")
            assert subvertical == "road bikes"
            
            # Verify correct LLM call
            mock_chat.assert_called_once()
            call_args = mock_chat.call_args[1]
            assert call_args["task"] == "brand_research"
            assert "road bikes" in call_args["messages"][0]["content"] or "cycling" in call_args["messages"][0]["content"]
    
    @pytest.mark.asyncio  
    async def test_detect_vertical_context_complete(self):
        """Test complete vertical context detection"""
        with patch('src.llm.simple_factory.LLMFactory.chat_completion') as mock_chat:
            # Mock responses for brand and product detection
            mock_chat.side_effect = [
                {"content": "cycling", "usage": {"total_tokens": 20}},  # Brand vertical
                {"content": "mountain bikes", "usage": {"total_tokens": 25}}  # Product subvertical
            ]
            
            generator = DescriptorGenerator()
            
            product = Product(
                id="mtb-1",
                name="Mountain Bike Trail",
                brand="specialized.com",
                categories=["Mountain Bikes"]
            )
            
            context = await generator.detect_vertical_context(product)
            
            assert context["brand_vertical"] == "cycling"
            assert context["product_subvertical"] == "mountain bikes"
            assert context["effective_vertical"] == "mountain bikes"
            assert mock_chat.call_count == 2


class TestPromptGeneration:
    """Test prompt generation for different scenarios"""
    
    def test_build_descriptor_prompt_cycling(self):
        """Test descriptor prompt for cycling products"""
        generator = DescriptorGenerator()
        
        cycling_product = Product(
            id="bike-1",
            name="Road Bike",
            brand="TestBrand",
            categories=["Cycling"],
            originalPrice="$2000"
        )
        
        prompt = generator.build_descriptor_prompt(cycling_product, "cycling")
        
        assert "cycling products" in prompt
        assert "Road Bike" in prompt
        assert "TestBrand" in prompt
        assert "$2000" in prompt
        assert "Guidelines:" in prompt
    
    def test_build_sizing_prompt_proven_instruction(self):
        """Test that sizing prompt uses the exact proven instruction"""
        generator = DescriptorGenerator()
        
        product = Product(id="test-1", name="Test Product")
        sizing_data = {"S": "Small fit", "M": "Medium fit"}
        
        prompt = generator.build_sizing_prompt(product, sizing_data)
        
        # Verify the EXACT proven sizing instruction is used
        proven_instruction = "Given these product details and the sizing chart, find the correct sizing and create a 'sizing' field with the appropriate size information in JSON format."
        assert proven_instruction in prompt
        
        # Verify product details are included
        assert "Test Product" in prompt
        assert json.dumps(sizing_data, indent=2) in prompt
        
        # Verify example format is provided
        assert "Example response format:" in prompt
        assert '"sizing"' in prompt


class TestDescriptorGeneration:
    """Test descriptor generation functionality"""
    
    @pytest.mark.asyncio
    async def test_generate_descriptor_success(self):
        """Test successful descriptor generation"""
        with patch('src.llm.simple_factory.LLMFactory.chat_completion') as mock_chat:
            # Mock brand vertical detection
            mock_chat.side_effect = [
                {"content": "cycling", "usage": {"total_tokens": 20}},  # Brand vertical
                {"content": "none", "usage": {"total_tokens": 15}},     # Product subvertical
                {  # Descriptor generation
                    "content": "This high-performance road bike features a lightweight carbon frame and professional-grade components.",
                    "usage": {"total_tokens": 150},
                    "model": "gpt-4-turbo"
                }
            ]
            
            generator = DescriptorGenerator()
            
            product = Product(
                id="bike-1",
                name="Road Bike Pro",
                brand="specialized.com",
                categories=["Cycling"]
            )
            
            descriptor = await generator.generate_descriptor(product)
            
            assert descriptor is not None
            assert "high-performance road bike" in descriptor
            assert "carbon frame" in descriptor
            
            # Verify correct LLM calls
            assert mock_chat.call_count == 3
            # Last call should be descriptor generation
            final_call = mock_chat.call_args
            assert final_call[1]["task"] == "descriptor_generation"
            assert final_call[1]["temperature"] == 0.7  # Creative temperature
    
    @pytest.mark.asyncio
    async def test_generate_descriptor_empty_response(self):
        """Test handling of empty LLM response"""
        with patch('src.llm.simple_factory.LLMFactory.chat_completion') as mock_chat:
            mock_chat.side_effect = [
                {"content": "general", "usage": {"total_tokens": 10}},  # Brand vertical
                {"content": "none", "usage": {"total_tokens": 10}},     # Product subvertical
                {"content": "", "usage": {"total_tokens": 0}}           # Empty descriptor
            ]
            
            generator = DescriptorGenerator()
            product = Product(id="test-1", name="Test Product", brand="test.com")
            
            descriptor = await generator.generate_descriptor(product)
            assert descriptor is None
    
    @pytest.mark.asyncio
    async def test_generate_descriptor_llm_error(self):
        """Test handling of LLM errors"""
        with patch('src.llm.simple_factory.LLMFactory.chat_completion') as mock_chat:
            mock_chat.side_effect = [
                {"content": "general", "usage": {"total_tokens": 10}},  # Brand vertical
                {"content": "none", "usage": {"total_tokens": 10}},     # Product subvertical  
                TokenLimitError("Token limit exceeded")                 # Descriptor error
            ]
            
            generator = DescriptorGenerator()
            product = Product(id="test-1", name="Test Product", brand="test.com")
            
            descriptor = await generator.generate_descriptor(product)
            assert descriptor is None


class TestSizingGeneration:
    """Test sizing generation functionality"""
    
    @pytest.mark.asyncio
    async def test_generate_sizing_success(self):
        """Test successful sizing generation"""
        with patch('src.llm.simple_factory.LLMFactory.chat_completion') as mock_chat:
            # Mock successful sizing response
            sizing_response = {
                "sizing": {
                    "size_chart": {
                        "S": "Fits 34-36 inch chest",
                        "M": "Fits 38-40 inch chest",
                        "L": "Fits 42-44 inch chest"
                    },
                    "fit_advice": "This product runs true to size. Measure chest at fullest part."
                }
            }
            
            mock_chat.return_value = {
                "content": json.dumps(sizing_response),
                "usage": {"total_tokens": 200},
                "model": "gpt-4"
            }
            
            generator = DescriptorGenerator()
            
            product = Product(
                id="shirt-1",
                name="Cotton T-Shirt",
                sizes=["S", "M", "L"]
            )
            
            sizing_data = {
                "S": "Small",
                "M": "Medium", 
                "L": "Large"
            }
            
            sizing = await generator.generate_sizing(product, sizing_data)
            
            assert sizing is not None
            assert "sizing" in sizing
            assert "size_chart" in sizing["sizing"]
            assert "S" in sizing["sizing"]["size_chart"]
            assert "fit_advice" in sizing["sizing"]
            
            # Verify correct LLM call
            mock_chat.assert_called_once()
            call_args = mock_chat.call_args[1]
            assert call_args["task"] == "sizing_analysis"
            assert call_args["temperature"] == 0.3  # Accuracy temperature
    
    @pytest.mark.asyncio
    async def test_generate_sizing_json_extraction(self):
        """Test JSON extraction from wrapped response"""
        with patch('src.llm.simple_factory.LLMFactory.chat_completion') as mock_chat:
            # Mock response with JSON wrapped in text
            sizing_json = {"sizing": {"size_chart": {"S": "Small fit"}}}
            wrapped_response = f"Here's the sizing information:\n{json.dumps(sizing_json)}\nHope this helps!"
            
            mock_chat.return_value = {
                "content": wrapped_response,
                "usage": {"total_tokens": 150}
            }
            
            generator = DescriptorGenerator()
            product = Product(id="test-1")
            sizing_data = {"S": "Small"}
            
            sizing = await generator.generate_sizing(product, sizing_data)
            
            assert sizing is not None
            assert sizing == sizing_json
    
    @pytest.mark.asyncio
    async def test_generate_sizing_invalid_json(self):
        """Test handling of invalid JSON response"""
        with patch('src.llm.simple_factory.LLMFactory.chat_completion') as mock_chat:
            mock_chat.return_value = {
                "content": "This is not valid JSON content",
                "usage": {"total_tokens": 50}
            }
            
            generator = DescriptorGenerator()
            product = Product(id="test-1")
            sizing_data = {"S": "Small"}
            
            sizing = await generator.generate_sizing(product, sizing_data)
            assert sizing is None


class TestProductProcessing:
    """Test end-to-end product processing"""
    
    @pytest.mark.asyncio
    async def test_process_product_complete(self):
        """Test complete product processing with descriptor and sizing"""
        with patch('src.llm.simple_factory.LLMFactory.chat_completion') as mock_chat:
            # Mock responses for vertical detection, descriptor, and sizing
            mock_chat.side_effect = [
                {"content": "cycling", "usage": {"total_tokens": 20}},     # Brand vertical
                {"content": "none", "usage": {"total_tokens": 15}},        # Product subvertical
                {"content": "Great cycling product with excellent features.", "usage": {"total_tokens": 100}},  # Descriptor
                {"content": json.dumps({"sizing": {"size_chart": {"S": "Small frame"}}}), "usage": {"total_tokens": 150}}  # Sizing
            ]
            
            generator = DescriptorGenerator()
            
            product = Product(
                id="bike-1",
                name="Test Bike",
                brand="specialized.com",
                categories=["Cycling"]
            )
            
            sizing_data = {"S": "Small", "M": "Medium"}
            
            results = await generator.process_product(product, sizing_data)
            
            assert results["product_id"] == "bike-1"
            assert results["product_name"] == "Test Bike"
            assert results["detected_vertical"] == "cycling"
            assert results["descriptor"] is not None
            assert results["sizing"] is not None
            assert results["processing_time"] is not None
            assert results["processing_time"] > 0
            assert len(results["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_process_product_no_sizing_data(self):
        """Test product processing without sizing data"""
        with patch('src.llm.simple_factory.LLMFactory.chat_completion') as mock_chat:
            mock_chat.side_effect = [
                {"content": "general", "usage": {"total_tokens": 20}},     # Brand vertical
                {"content": "none", "usage": {"total_tokens": 15}},        # Product subvertical
                {"content": "Great product description.", "usage": {"total_tokens": 100}}  # Descriptor
            ]
            
            generator = DescriptorGenerator()
            product = Product(id="test-1", name="Test Product", brand="test.com")
            
            results = await generator.process_product(product)  # No sizing data
            
            assert results["descriptor"] is not None
            assert results["sizing"] is None  # No sizing generated
            assert len(results["errors"]) == 0


class TestFactoryFunctions:
    """Test factory and convenience functions"""
    
    def test_get_descriptor_generator(self):
        """Test descriptor generator factory function"""
        generator = get_descriptor_generator()
        
        assert isinstance(generator, DescriptorGenerator)
        assert generator.settings is not None
    
    @pytest.mark.asyncio
    async def test_generate_product_descriptor_convenience(self):
        """Test convenience function for descriptor generation"""
        with patch('src.descriptor.get_descriptor_generator') as mock_factory:
            mock_generator = AsyncMock()
            mock_generator.generate_descriptor.return_value = "Test descriptor"
            mock_factory.return_value = mock_generator
            
            product = Product(id="test-1", name="Test Product")
            
            descriptor = await generate_product_descriptor(product)
            
            assert descriptor == "Test descriptor"
            mock_generator.generate_descriptor.assert_called_once_with(product)
    
    @pytest.mark.asyncio
    async def test_generate_product_sizing_convenience(self):
        """Test convenience function for sizing generation"""
        with patch('src.descriptor.get_descriptor_generator') as mock_factory:
            mock_generator = AsyncMock()
            mock_sizing = {"sizing": {"size_chart": {"S": "Small"}}}
            mock_generator.generate_sizing.return_value = mock_sizing
            mock_factory.return_value = mock_generator
            
            product = Product(id="test-1", name="Test Product")
            sizing_data = {"S": "Small"}
            
            sizing = await generate_product_sizing(product, sizing_data)
            
            assert sizing == mock_sizing
            mock_generator.generate_sizing.assert_called_once_with(product, sizing_data)


class TestConfigurationIntegration:
    """Test integration with configuration system"""
    
    def test_generator_uses_settings(self):
        """Test that generator properly uses configuration settings"""
        with patch('src.descriptor.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.openai_max_tokens = 1500
            mock_get_settings.return_value = mock_settings
            
            generator = DescriptorGenerator()
            
            assert generator.settings == mock_settings
            assert generator.settings.openai_max_tokens == 1500


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_handle_token_limit_error(self):
        """Test handling of token limit errors"""
        with patch('src.llm.simple_factory.LLMFactory.chat_completion') as mock_chat:
            mock_chat.side_effect = [
                {"content": "general", "usage": {"total_tokens": 20}},     # Brand vertical
                {"content": "none", "usage": {"total_tokens": 15}},        # Product subvertical
                TokenLimitError("Token limit exceeded")                    # Descriptor error
            ]
            
            generator = DescriptorGenerator()
            product = Product(id="test-1", name="Test Product", brand="test.com")
            
            descriptor = await generator.generate_descriptor(product)
            assert descriptor is None
    
    @pytest.mark.asyncio
    async def test_handle_model_not_found_error(self):
        """Test handling of model not found errors"""
        with patch('src.llm.simple_factory.LLMFactory.chat_completion') as mock_chat:
            mock_chat.side_effect = [
                {"content": "general", "usage": {"total_tokens": 20}},     # Brand vertical
                {"content": "none", "usage": {"total_tokens": 15}},        # Product subvertical
                ModelNotFoundError("Model not found")                      # Descriptor error
            ]
            
            generator = DescriptorGenerator()
            product = Product(id="test-1", name="Test Product", brand="test.com")
            
            descriptor = await generator.generate_descriptor(product)
            assert descriptor is None
    
    @pytest.mark.asyncio
    async def test_handle_general_exception(self):
        """Test handling of general exceptions"""
        with patch('src.llm.simple_factory.LLMFactory.chat_completion') as mock_chat:
            mock_chat.side_effect = [
                {"content": "general", "usage": {"total_tokens": 20}},     # Brand vertical
                {"content": "none", "usage": {"total_tokens": 15}},        # Product subvertical
                Exception("Unexpected error")                              # Descriptor error
            ]
            
            generator = DescriptorGenerator()
            product = Product(id="test-1", name="Test Product", brand="test.com")
            
            descriptor = await generator.generate_descriptor(product)
            assert descriptor is None


# Integration test fixtures for real testing (uncomment when running with real API)
"""
@pytest.mark.integration
class TestRealLLMIntegration:
    # Real integration tests - requires API keys and environment setup
    
    @pytest.mark.asyncio
    async def test_real_descriptor_generation(self):
        # Test with real OpenAI API - requires valid API key
        generator = get_descriptor_generator()
        
        product = Product(
            id="real-test-1",
            name="Specialized Tarmac SL7",
            brand="Specialized",
            categories=["Road Bikes", "Performance"],
            originalPrice="$4,000",
            highlights=["Carbon Frame", "Lightweight", "Aerodynamic"]
        )
        
        descriptor = await generator.generate_descriptor(product)
        
        assert descriptor is not None
        assert len(descriptor) > 50
        assert "Tarmac" in descriptor or "road bike" in descriptor.lower()
    
    @pytest.mark.asyncio
    async def test_real_sizing_generation(self):
        # Test with real OpenAI API - requires valid API key
        generator = get_descriptor_generator()
        
        product = Product(
            id="real-test-2",
            name="Cotton T-Shirt",
            categories=["Clothing"],
            sizes=["S", "M", "L", "XL"]
        )
        
        sizing_data = {
            "S": "Chest: 34-36 inches",
            "M": "Chest: 38-40 inches",
            "L": "Chest: 42-44 inches",
            "XL": "Chest: 46-48 inches"
        }
        
        sizing = await generator.generate_sizing(product, sizing_data)
        
        assert sizing is not None
        assert "sizing" in sizing
        assert isinstance(sizing["sizing"], dict)
""" 