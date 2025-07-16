"""Manager for orchestrating descriptor enhancement modules."""

import logging
from typing import List, Dict, Any, Optional, Type
from liddy.models.product import Product
from .base import BaseDescriptorModule
from .modules import PriceModule, SaleModule, VariantModule

logger = logging.getLogger(__name__)


class DescriptorModuleManager:
    """Manages and orchestrates descriptor enhancement modules."""
    
    # Default modules to load
    DEFAULT_MODULES = [
        PriceModule,
        SaleModule,
        VariantModule
    ]
    
    def __init__(self, modules: Optional[List[Type[BaseDescriptorModule]]] = None, 
                 global_config: Optional[Dict[str, Any]] = None):
        """Initialize the module manager.
        
        Args:
            modules: List of module classes to use. If None, uses DEFAULT_MODULES.
            global_config: Global configuration passed to all modules.
        """
        self.global_config = global_config or {}
        self.modules: List[BaseDescriptorModule] = []
        
        # Load modules
        module_classes = modules or self.DEFAULT_MODULES
        for module_class in module_classes:
            try:
                # Get module-specific config if available
                module_config = self.global_config.get(f'{module_class.__name__.lower()}_config', {})
                # Merge with global config
                config = {**self.global_config, **module_config}
                
                module = module_class(config=config)
                self.modules.append(module)
                logger.debug(f"Loaded module: {module.name}")
            except Exception as e:
                logger.error(f"Failed to load module {module_class.__name__}: {e}")
        
        # Sort modules by priority
        self.modules.sort(key=lambda m: m.priority)
        logger.info(f"Initialized DescriptorModuleManager with {len(self.modules)} modules")
    
    def enhance_product(self, product: Product, initial_descriptor: str, **kwargs) -> Dict[str, Any]:
        """Enhance a product using all applicable modules.
        
        Args:
            product: Product to enhance
            initial_descriptor: Initial descriptor text
            **kwargs: Additional context passed to modules (e.g., price_stats)
            
        Returns:
            Dict containing:
                - descriptor: Enhanced descriptor text
                - search_keywords: Enhanced search keywords
                - metadata: Combined metadata from all modules
                - modules_applied: List of module names that were applied
        """
        descriptor = initial_descriptor
        search_keywords = product.search_keywords or []
        combined_metadata = {}
        modules_applied = []
        
        # Run each module
        for module in self.modules:
            try:
                if not module.is_applicable(product):
                    logger.debug(f"Module {module.name} not applicable to product {product.id}")
                    continue
                
                # Enhance descriptor
                descriptor = module.enhance_descriptor(descriptor, product, **kwargs)
                
                # Enhance search keywords
                search_keywords = module.enhance_search_keywords(search_keywords, product, **kwargs)
                
                # Collect metadata
                module_metadata = module.get_metadata(product, **kwargs)
                if module_metadata:
                    combined_metadata[f'{module.name}_metadata'] = module_metadata
                
                modules_applied.append(module.name)
                logger.debug(f"Applied module {module.name} to product {product.id}")
                
            except Exception as e:
                logger.error(f"Error applying module {module.name} to product {product.id}: {e}")
        
        # Deduplicate and limit search keywords
        seen = set()
        unique_keywords = []
        for keyword in search_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in seen:
                seen.add(keyword_lower)
                unique_keywords.append(keyword)
        
        # Limit to reasonable number
        unique_keywords = unique_keywords[:50]
        
        return {
            'descriptor': descriptor.strip(),
            'search_keywords': unique_keywords,
            'metadata': combined_metadata,
            'modules_applied': modules_applied
        }
    
    def add_module(self, module_class: Type[BaseDescriptorModule], config: Optional[Dict[str, Any]] = None):
        """Add a new module to the manager.
        
        Args:
            module_class: Module class to instantiate and add
            config: Optional module-specific configuration
        """
        try:
            module_config = {**self.global_config, **(config or {})}
            module = module_class(config=module_config)
            self.modules.append(module)
            
            # Re-sort by priority
            self.modules.sort(key=lambda m: m.priority)
            
            logger.info(f"Added module: {module.name}")
        except Exception as e:
            logger.error(f"Failed to add module {module_class.__name__}: {e}")
    
    def remove_module(self, module_name: str) -> bool:
        """Remove a module by name.
        
        Args:
            module_name: Name of the module to remove
            
        Returns:
            True if module was removed, False otherwise
        """
        initial_count = len(self.modules)
        self.modules = [m for m in self.modules if m.name != module_name]
        
        if len(self.modules) < initial_count:
            logger.info(f"Removed module: {module_name}")
            return True
        
        return False
    
    def get_module(self, module_name: str) -> Optional[BaseDescriptorModule]:
        """Get a module by name.
        
        Args:
            module_name: Name of the module to retrieve
            
        Returns:
            Module instance or None if not found
        """
        for module in self.modules:
            if module.name == module_name:
                return module
        return None
    
    def list_modules(self) -> List[Dict[str, Any]]:
        """List all loaded modules with their info.
        
        Returns:
            List of module info dicts
        """
        return [
            {
                'name': module.name,
                'priority': module.priority,
                'class': module.__class__.__name__
            }
            for module in self.modules
        ]