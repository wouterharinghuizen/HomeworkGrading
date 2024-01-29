#!/bin/bash
set -e

pip install -r requirements_dev.txt
pip install -r requirements.txt
pip install -e .
pre-commit install
