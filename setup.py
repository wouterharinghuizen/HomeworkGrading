#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = []

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest",
]

setup(
    author="Wouter Haringhuizen",
    author_email="wouterharinghuizen@hotmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
    ],
    description="",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="handwrittenhomeworkgrading",
    name="handwrittenhomeworkgrading",
    packages=find_packages(include=["handwrittenhomeworkgrading"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://git.kpmg.nl/KPMG-NL-AABD/ResearchAndDevelopment/ImageAnalysis/handwrittenhomeworkgrading",
    version="0.1.0",
    zip_safe=False,
)
