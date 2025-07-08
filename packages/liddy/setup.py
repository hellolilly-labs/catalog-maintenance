from setuptools import setup, find_packages

setup(
    name="liddy",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0.0",
        "pydantic-settings>=2.1.0",
        "python-dotenv>=1.0.0",
        "google-cloud-storage>=2.10.0",
        "pinecone[grpc]>=7.3.0",
        "openai>=1.0.0",
        "anthropic>=0.55.0",
        "langfuse>=2.0.0",
        "redis>=4.0.0",
        "aiohttp>=3.8.0",
        "tiktoken>=0.5.0",
    ],
)