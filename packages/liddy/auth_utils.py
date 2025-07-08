"""
Authentication utilities for Google Cloud across different environments.

Handles authentication for:
1. Local IDE development (using gcloud auth or service account key)
2. Docker local testing (using mounted gcloud config or service account key)
3. Cloud Run deployment (using metadata service)
"""

import os
import logging
from typing import Optional
from google.auth import default
from google.auth.credentials import Credentials

logger = logging.getLogger(__name__)


def get_google_credentials() -> Optional[Credentials]:
    """
    Get Google Cloud credentials in a way that works across all environments.
    
    Order of precedence:
    1. GOOGLE_APPLICATION_CREDENTIALS environment variable (if set)
    2. gcloud application default credentials
    3. Cloud Run/GCE metadata service
    
    Returns:
        Google credentials object or None if authentication fails
    """
    try:
        # Let google-auth library handle the detection
        credentials, project = default()
        
        if credentials:
            logger.info(f"Successfully authenticated with Google Cloud (project: {project})")
            return credentials
        else:
            logger.warning("No Google Cloud credentials found")
            return None
            
    except Exception as e:
        logger.warning(f"Failed to get Google Cloud credentials: {e}")
        # Don't fail hard - allow the app to continue
        # Some operations might work without auth (e.g., public buckets)
        return None


def setup_google_auth():
    """
    Set up Google Cloud authentication for the current environment.
    
    This function checks the environment and provides helpful logging
    about which authentication method is being used.
    """
    # Check if we have GOOGLE_APPLICATION_CREDENTIALS
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        logger.info(f"Using service account key from GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
    
    # Check if we're running on Cloud Run
    elif os.getenv('K_SERVICE'):  # Cloud Run sets this
        logger.info("Running on Cloud Run - using metadata service for authentication")
    
    # Check if we have gcloud config
    elif os.path.exists(os.path.expanduser('~/.config/gcloud/application_default_credentials.json')):
        logger.info("Using gcloud application default credentials")
    
    # Check if we're in a container with mounted gcloud config
    elif os.path.exists('/home/appuser/.config/gcloud/application_default_credentials.json'):
        logger.info("Using mounted gcloud credentials in container")
    
    else:
        logger.warning("No Google Cloud authentication method detected. Some features may not work.")
        logger.info("To authenticate locally, run: gcloud auth application-default login")
    
    # Try to get credentials to verify they work
    creds = get_google_credentials()
    return creds is not None