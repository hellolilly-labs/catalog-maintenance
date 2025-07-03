#!/usr/bin/env python3
"""
Migrate research files from old structure to new structure in GCP Storage.

Old: <account>/research_phases/<researcher_name>_research.md
New: <account>/research/<researcher_name>/research.md

This script:
1. Lists all accounts in the bucket
2. For each account, moves research files from research_phases/ to research/<name>/
3. Preserves all metadata and sources files
"""

import os
import sys
import logging
from google.cloud import storage
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchStorageMigrator:
    def __init__(self, bucket_name: str = "liddy-account-documents-dev"):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
    def list_accounts(self) -> List[str]:
        """List all accounts in the bucket."""
        accounts = set()
        
        # List all blobs and extract account names
        for blob in self.bucket.list_blobs(prefix="accounts/"):
            parts = blob.name.split("/")
            if len(parts) >= 2 and parts[1]:
                accounts.add(parts[1])
        
        return sorted(list(accounts))
    
    def get_research_files(self, account: str) -> List[Tuple[str, str]]:
        """Get all research files for an account that need migration."""
        old_prefix = f"accounts/{account}/research_phases/"
        migrations = []
        
        for blob in self.bucket.list_blobs(prefix=old_prefix):
            old_path = blob.name
            filename = os.path.basename(old_path)
            
            # Extract researcher name from filename patterns
            researcher_name = None
            
            if filename.endswith("_research.md"):
                researcher_name = filename[:-12]  # Remove "_research.md"
                new_filename = "research.md"
            elif filename.endswith("_research_metadata.json"):
                researcher_name = filename[:-23]  # Remove "_research_metadata.json"
                new_filename = "research_metadata.json"
            elif filename.endswith("_research_sources.json"):
                researcher_name = filename[:-22]  # Remove "_research_sources.json"
                new_filename = "research_sources.json"
            else:
                logger.warning(f"Unknown file pattern: {filename}")
                continue
            
            if researcher_name:
                new_path = f"accounts/{account}/research/{researcher_name}/{new_filename}"
                migrations.append((old_path, new_path))
        
        return migrations
    
    def migrate_file(self, old_path: str, new_path: str) -> bool:
        """Copy a file from old path to new path in GCS."""
        try:
            source_blob = self.bucket.blob(old_path)
            if not source_blob.exists():
                logger.warning(f"Source file doesn't exist: {old_path}")
                return False
            
            # Copy to new location
            destination_blob = self.bucket.blob(new_path)
            destination_blob.upload_from_string(source_blob.download_as_text())
            
            logger.info(f"✅ Migrated: {old_path} → {new_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate {old_path}: {e}")
            return False
    
    def delete_old_file(self, old_path: str):
        """Delete the old file after successful migration."""
        try:
            blob = self.bucket.blob(old_path)
            blob.delete()
            logger.info(f"🗑️  Deleted old file: {old_path}")
        except Exception as e:
            logger.error(f"❌ Failed to delete {old_path}: {e}")
    
    def migrate_account(self, account: str, dry_run: bool = True, cleanup: bool = False) -> int:
        """Migrate all research files for a single account."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing account: {account}")
        
        migrations = self.get_research_files(account)
        
        if not migrations:
            logger.info(f"No research files to migrate for {account}")
            return 0
        
        logger.info(f"Found {len(migrations)} files to migrate")
        
        successful = 0
        cleanup_list = []
        for old_path, new_path in migrations:
            if dry_run:
                logger.info(f"[DRY RUN] Would migrate: {old_path} → {new_path}")
                successful += 1
            else:
                if self.migrate_file(old_path, new_path):
                    successful += 1
                    cleanup_list.append(old_path)
        
        # Cleanup old files if requested and not in dry run
        if cleanup and not dry_run and cleanup_list:
            logger.info(f"\n🧹 Cleaning up {len(cleanup_list)} old files...")
            for old_path in cleanup_list:
                self.delete_old_file(old_path)
        
        return successful
    
    def run_migration(self, dry_run: bool = True, cleanup: bool = False):
        """Run the full migration for all accounts."""
        logger.info(f"Starting research storage migration in bucket: {self.bucket_name}")
        if dry_run:
            logger.info("🔍 DRY RUN MODE - No files will be modified")
        elif cleanup:
            logger.info("🧹 CLEANUP MODE - Old files will be deleted after successful migration")
        
        accounts = self.list_accounts()
        logger.info(f"Found {len(accounts)} accounts to process")
        
        total_migrated = 0
        for account in accounts:
            migrated = self.migrate_account(account, dry_run, cleanup)
            total_migrated += migrated
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Migration complete!")
        logger.info(f"Total files {'would be' if dry_run else ''} migrated: {total_migrated}")
        
        if dry_run:
            logger.info("\nTo perform the actual migration, run with --execute flag")
            if not cleanup:
                logger.info("To also cleanup old files, add --cleanup flag")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate research storage structure in GCS")
    parser.add_argument("--execute", action="store_true", 
                       help="Execute the migration (default is dry run)")
    parser.add_argument("--cleanup", action="store_true",
                       help="Delete old files after successful migration (requires --execute)")
    parser.add_argument("--bucket", default="liddy-account-documents-dev",
                       help="GCS bucket name (default: liddy-account-documents-dev)")
    parser.add_argument("--account", help="Migrate only a specific account")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.cleanup and not args.execute:
        parser.error("--cleanup requires --execute flag")
    
    # Ensure Google Cloud credentials are set
    settings = get_settings()
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(settings.GOOGLE_APPLICATION_CREDENTIALS)
    
    migrator = ResearchStorageMigrator(args.bucket)
    
    if args.account:
        # Migrate single account
        migrator.migrate_account(args.account, dry_run=not args.execute, cleanup=args.cleanup)
    else:
        # Migrate all accounts
        migrator.run_migration(dry_run=not args.execute, cleanup=args.cleanup)


if __name__ == "__main__":
    main()