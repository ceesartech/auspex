from setuptools import find_packages, setup

setup(
    name="betting-system",
    version="1.0.0",
    description="Personal Sports Betting & Lottery Recommendation System",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        # Read from requirements.txt
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    extras_require={
        "dev": [line.strip() for line in open("requirements-dev.txt") if line.strip() and not line.startswith("#")],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
    ],
)
