import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file

class ConfigService:
    @staticmethod
    def get(key: str, default=None):
        return os.environ[key] if key in os.environ else default
        # return os.getenv(key, default)

# # Example usage
# config = ConfigService()
# db_host = config.get("DATABASE_HOST", "localhost")
# print(db_host)