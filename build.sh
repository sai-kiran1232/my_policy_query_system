#!/bin/bash

echo "Installing spaCy and downloading model..."

# Step 1: Install spaCy
pip install spacy

# Step 2: Download English spaCy model
python -m spacy download en_core_web_sm

echo "Done setting up spaCy."
