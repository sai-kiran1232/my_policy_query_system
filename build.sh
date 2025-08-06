#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Installing spaCy model..."
python -m spacy download en_core_web_sm
