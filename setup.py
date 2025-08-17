from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sales-agent",
    version="0.1.0",
    author="Your Name",
    author_email="sayalisonawane27@gmail.com",
    description="A sales agent project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sayali-sonawane/ads-sales-agent",
    packages=find_packages(),
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Add your project dependencies here
    ],
)
