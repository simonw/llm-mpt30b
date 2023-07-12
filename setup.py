from setuptools import setup
import os

VERSION = "0.1"


def get_long_description():
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
        encoding="utf8",
    ) as fp:
        return fp.read()


setup(
    name="llm-mpt30b",
    description="Plugin for LLM adding support for the MPT-30B language model",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Simon Willison",
    url="https://github.com/simonw/llm-mpt30b",
    project_urls={
        "Issues": "https://github.com/simonw/llm-mpt30b/issues",
        "CI": "https://github.com/simonw/llm-mpt30b/actions",
        "Changelog": "https://github.com/simonw/llm-mpt30b/releases",
    },
    license="Apache License, Version 2.0",
    classifiers=["License :: OSI Approved :: Apache Software License"],
    version=VERSION,
    modules=["llm_mpt30b"],
    entry_points={"llm": ["llm_mpt30b = llm_mpt30b"]},
    install_requires=[
        "llm>=0.5",
        "ctransformers>=0.2.10",
        "transformers>=4.30.2",
        "huggingface-hub",
    ],
    extras_require={"test": ["pytest"]},
    python_requires=">=3.9",
)
