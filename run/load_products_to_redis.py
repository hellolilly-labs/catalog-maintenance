#!/usr/bin/env python3
"""
DEPRECATED: This functionality has been moved to voice_agent.py

Use: python packages/liddy_voice/voice_agent.py load-data

This script is kept for backwards compatibility only.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from liddy.models.redis_product_loader import RedisProductLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Redirect to new command."""
    logger.warning("⚠️  This script is deprecated!")
    logger.info("Please use: python packages/liddy_voice/voice_agent.py load-data")
    
    # For backwards compatibility, still run the loader
    accounts = sys.argv[1:] if len(sys.argv) > 1 else None
    
    loader = RedisProductLoader()
    try:
        await loader.connect()
        results = await loader.load_all_accounts(accounts)
        
        # Verify loading for each account
        for account in results:
            if results[account] > 0:
                verified = await loader.verify_loading(account)
                if not verified:
                    logger.error(f"❌ Verification failed for {account}")
                    sys.exit(1)
        
        logger.info("✅ Products loaded (please switch to new command)")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        await loader.disconnect()


if __name__ == "__main__":
    asyncio.run(main())