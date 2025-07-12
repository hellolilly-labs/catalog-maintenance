#\!/usr/bin/env python3
"""
Setup configuration for liddy_voice package
"""

from setuptools import setup

setup(
    name="liddy_voice",
    version="0.1.0",
    description="Voice assistant components for Liddy AI",
    author="Liddy AI",
    packages=["liddy_voice", "liddy_voice.conversation"],
    package_dir={"liddy_voice": "."},
    python_requires=">=3.12",
    install_requires=[
        "liddy",  # Core package dependency
        "livekit",
        "livekit-agents",
        "aiofiles",
        "redis",
        "pinecone>=6.0.2",
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