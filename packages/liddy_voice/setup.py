#\!/usr/bin/env python3
"""
Setup configuration for liddy_voice package
"""

from setuptools import setup, find_packages

setup(
    name="liddy_voice",
    version="0.1.0",
    description="Voice assistant components for Liddy AI",
    author="Liddy AI",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "liddy",  # Core package dependency
        "livekit",
        "livekit-agents",
        "aiofiles",
        "redis",
        "pinecone-client",
        "python-dotenv",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-asyncio",
            "black",
            "flake8",
        ],
    },
)
EOF < /dev/null