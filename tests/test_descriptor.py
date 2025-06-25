"""
Test suite for descriptor generation system

Tests for LLM-powered descriptor and sizing generation including:
- Proven sizing instruction implementation
- Vertical auto-detection capabilities
- OpenAI service integration
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
    """Test the main DescriptorGenerator class"""
    
    @pytest.fixture
    def generator(self):
        """Create a test descriptor generator with mocked LLM router"""
        with patch('src.descriptor.LLMRouter') as mock_router_class:
            mock_router = AsyncMock()
            mock_router_class.return_value = mock_router
            generator = DescriptorGenerator()
            generator.llm_router = mock_router
            return generator, mock_router
    
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
    """Test vertical auto-detection functionality"""
    
    def test_detect_cycling_vertical(self):
        """Test detection of cycling products"""
        generator = DescriptorGenerator()
        
        cycling_product = Product(
            id="bike-1",
            name="Mountain Bike Pro",
            categories=["Cycling", "Mountain Bikes"],
            highlights=["Carbon frame", "Shimano gears"]
        )
        
        vertical = generator.detect_vertical(cycling_product)
        assert vertical == "cycling"
    
    def test_detect_fashion_vertical(self):
        """Test detection of fashion products"""
        generator = DescriptorGenerator()
        
        fashion_product = Product(
            id="shirt-1",
            name="Designer Shirt",
            categories=["Clothing", "Apparel"],
            highlights=["Cotton blend", "Stylish design"]
        )
        
        vertical = generator.detect_vertical(fashion_product)
        assert vertical == "fashion"
    
    def test_detect_electronics_vertical(self):
        """Test detection of electronics products"""
        generator = DescriptorGenerator()
        
        electronics_product = Product(
            id="device-1",
            name="Smart Phone",
            categories=["Electronics", "Mobile"],
            highlights=["Digital camera", "Battery life", "Processor"]
        )
        
        vertical = generator.detect_vertical(electronics_product)
        assert vertical == "electronics"
    
    def test_detect_general_vertical_fallback(self):
        """Test fallback to general for unclear products"""
        generator = DescriptorGenerator()
        
        unclear_product = Product(
            id="product-1",
            name="Mystery Item",
            categories=["Unknown"],
            highlights=["Some feature"]
        )
        
        vertical = generator.detect_vertical(unclear_product)
        assert vertical == "general"
    
    def test_detect_vertical_with_empty_product(self):
        """Test vertical detection with minimal product data"""
        generator = DescriptorGenerator()
        
        minimal_product = Product(id="minimal-1")
        
        vertical = generator.detect_vertical(minimal_product)
        assert vertical == "general"


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
        with patch('src.descriptor.LLMRouter') as mock_router_class:
            mock_router = AsyncMock()
            mock_router_class.return_value = mock_router
            
            # Mock successful LLM response
            mock_router.chat_completion.return_value = {
                "content": "This high-performance road bike features a lightweight carbon frame and professional-grade components.",
                "usage": {"total_tokens": 150},
                "model": "gpt-4-turbo"
            }
            
            generator = DescriptorGenerator()
            generator.llm_router = mock_router
            
            product = Product(
                id="bike-1",
                name="Road Bike Pro",
                categories=["Cycling"]
            )
            
            descriptor = await generator.generate_descriptor(product)
            
            assert descriptor is not None
            assert "high-performance road bike" in descriptor
            assert "carbon frame" in descriptor
            
            # Verify correct LLM call
            mock_router.chat_completion.assert_called_once()
            call_args = mock_router.chat_completion.call_args
            assert call_args[1]["task"] == "descriptor_generation"
            assert call_args[1]["temperature"] == 0.7  # Creative temperature
    
    @pytest.mark.asyncio
    async def test_generate_descriptor_empty_response(self):
        """Test handling of empty LLM response"""
        with patch('src.descriptor.LLMRouter') as mock_router_class:
            mock_router = AsyncMock()
            mock_router_class.return_value = mock_router
            
            # Mock empty response
            mock_router.chat_completion.return_value = {
                "content": "",
                "usage": {"total_tokens": 0}
            }
            
            generator = DescriptorGenerator()
            generator.llm_router = mock_router
            
            product = Product(id="test-1", name="Test Product")
            
            descriptor = await generator.generate_descriptor(product)
            
            assert descriptor is None
    
    @pytest.mark.asyncio
    async def test_generate_descriptor_llm_error(self):
        """Test handling of LLM errors"""
        with patch('src.descriptor.LLMRouter') as mock_router_class:
            mock_router = AsyncMock()
            mock_router_class.return_value = mock_router
            
            # Mock LLM error
            mock_router.chat_completion.side_effect = TokenLimitError("Token limit exceeded")
            
            generator = DescriptorGenerator()
            generator.llm_router = mock_router
            
            product = Product(id="test-1", name="Test Product")
            
            descriptor = await generator.generate_descriptor(product)
            
            assert descriptor is None


class TestSizingGeneration:
    """Test sizing generation functionality"""
    
    @pytest.mark.asyncio
    async def test_generate_sizing_success(self):
        """Test successful sizing generation"""
        with patch('src.descriptor.LLMRouter') as mock_router_class:
            mock_router = AsyncMock()
            mock_router_class.return_value = mock_router
            
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
            
            mock_router.chat_completion.return_value = {
                "content": json.dumps(sizing_response),
                "usage": {"total_tokens": 200},
                "model": "gpt-4"
            }
            
            generator = DescriptorGenerator()
            generator.llm_router = mock_router
            
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
            mock_router.chat_completion.assert_called_once()
            call_args = mock_router.chat_completion.call_args
            assert call_args[1]["task"] == "sizing_analysis"
            assert call_args[1]["temperature"] == 0.3  # Accuracy temperature
    
    @pytest.mark.asyncio
    async def test_generate_sizing_json_extraction(self):
        """Test JSON extraction from wrapped response"""
        with patch('src.descriptor.LLMRouter') as mock_router_class:
            mock_router = AsyncMock()
            mock_router_class.return_value = mock_router
            
            # Mock response with JSON wrapped in text
            sizing_json = {"sizing": {"size_chart": {"S": "Small fit"}}}
            wrapped_response = f"Here's the sizing information:\n{json.dumps(sizing_json)}\nHope this helps!"
            
            mock_router.chat_completion.return_value = {
                "content": wrapped_response,
                "usage": {"total_tokens": 150}
            }
            
            generator = DescriptorGenerator()
            generator.llm_router = mock_router
            
            product = Product(id="test-1")
            sizing_data = {"S": "Small"}
            
            sizing = await generator.generate_sizing(product, sizing_data)
            
            assert sizing is not None
            assert sizing == sizing_json
    
    @pytest.mark.asyncio
    async def test_generate_sizing_invalid_json(self):
        """Test handling of invalid JSON response"""
        with patch('src.descriptor.LLMRouter') as mock_router_class:
            mock_router = AsyncMock()
            mock_router_class.return_value = mock_router
            
            # Mock invalid JSON response
            mock_router.chat_completion.return_value = {
                "content": "This is not valid JSON content",
                "usage": {"total_tokens": 50}
            }
            
            generator = DescriptorGenerator()
            generator.llm_router = mock_router
            
            product = Product(id="test-1")
            sizing_data = {"S": "Small"}
            
            sizing = await generator.generate_sizing(product, sizing_data)
            
            assert sizing is None


class TestProductProcessing:
    """Test end-to-end product processing"""
    
    @pytest.mark.asyncio
    async def test_process_product_complete(self):
        """Test complete product processing with descriptor and sizing"""
        with patch('src.descriptor.LLMRouter') as mock_router_class:
            mock_router = AsyncMock()
            mock_router_class.return_value = mock_router
            
            # Mock responses for both descriptor and sizing
            mock_router.chat_completion.side_effect = [
                {  # Descriptor response
                    "content": "Great cycling product with excellent features.",
                    "usage": {"total_tokens": 100}
                },
                {  # Sizing response
                    "content": json.dumps({"sizing": {"size_chart": {"S": "Small frame"}}}),
                    "usage": {"total_tokens": 150}
                }
            ]
            
            generator = DescriptorGenerator()
            generator.llm_router = mock_router
            
            product = Product(
                id="bike-1",
                name="Test Bike",
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
        with patch('src.descriptor.LLMRouter') as mock_router_class:
            mock_router = AsyncMock()
            mock_router_class.return_value = mock_router
            
            # Mock descriptor response only
            mock_router.chat_completion.return_value = {
                "content": "Great product description.",
                "usage": {"total_tokens": 100}
            }
            
            generator = DescriptorGenerator()
            generator.llm_router = mock_router
            
            product = Product(id="test-1", name="Test Product")
            
            results = await generator.process_product(product)  # No sizing data
            
            assert results["descriptor"] is not None
            assert results["sizing"] is None  # No sizing generated
            assert len(results["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_process_product_with_errors(self):
        """Test product processing with partial failures"""
        with patch('src.descriptor.LLMRouter') as mock_router_class:
            mock_router = AsyncMock()
            mock_router_class.return_value = mock_router
            
            # Mock descriptor success, sizing failure
            mock_router.chat_completion.side_effect = [
                {  # Descriptor success
                    "content": "Good product description.",
                    "usage": {"total_tokens": 100}
                },
                LLMError("Sizing generation failed")  # Sizing failure
            ]
            
            generator = DescriptorGenerator()
            generator.llm_router = mock_router
            
            product = Product(id="test-1", name="Test Product")
            sizing_data = {"S": "Small"}
            
            results = await generator.process_product(product, sizing_data)
            
            assert results["descriptor"] is not None
            assert results["sizing"] is None
            assert len(results["errors"]) == 1
            assert "Sizing generation failed" in results["errors"][0]


class TestFactoryFunctions:
    """Test factory and convenience functions"""
    
    def test_get_descriptor_generator(self):
        """Test descriptor generator factory function"""
        generator = get_descriptor_generator()
        
        assert isinstance(generator, DescriptorGenerator)
        assert generator.llm_router is not None
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
    """Test comprehensive error handling"""
    
    @pytest.mark.asyncio
    async def test_handle_token_limit_error(self):
        """Test handling of token limit errors"""
        with patch('src.descriptor.LLMRouter') as mock_router_class:
            mock_router = AsyncMock()
            mock_router_class.return_value = mock_router
            mock_router.chat_completion.side_effect = TokenLimitError("Token limit exceeded")
            
            generator = DescriptorGenerator()
            generator.llm_router = mock_router
            
            product = Product(id="test-1", name="Test Product")
            
            descriptor = await generator.generate_descriptor(product)
            assert descriptor is None
    
    @pytest.mark.asyncio
    async def test_handle_model_not_found_error(self):
        """Test handling of model not found errors"""
        with patch('src.descriptor.LLMRouter') as mock_router_class:
            mock_router = AsyncMock()
            mock_router_class.return_value = mock_router
            mock_router.chat_completion.side_effect = ModelNotFoundError("Model not available")
            
            generator = DescriptorGenerator()
            generator.llm_router = mock_router
            
            product = Product(id="test-1", name="Test Product")
            
            descriptor = await generator.generate_descriptor(product)
            assert descriptor is None
    
    @pytest.mark.asyncio
    async def test_handle_general_exception(self):
        """Test handling of unexpected exceptions"""
        with patch('src.descriptor.LLMRouter') as mock_router_class:
            mock_router = AsyncMock()
            mock_router_class.return_value = mock_router
            mock_router.chat_completion.side_effect = Exception("Unexpected error")
            
            generator = DescriptorGenerator()
            generator.llm_router = mock_router
            
            product = Product(id="test-1", name="Test Product")
            
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