#!/usr/bin/env python3
"""
Test script to verify Gemini API connectivity
"""

import google.generativeai as genai
import os

def test_gemini_api():
    # Test API key (replace with your actual key)
    api_key = "AIzaSyDdQBVpnm6-TGR--yuhVIGdBs8szQpkWzg"

    try:
        print("Testing Gemini API connection...")
        genai.configure(api_key=api_key)

        # First, list available models
        print("Listing available models...")
        models = genai.list_models()
        print("Available models:")
        for model in models:
            print(f"  - {model.name}")

        # Try with gemini-2.0-flash which should work
        print("Using model: gemini-2.0-flash")
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Hello, can you respond with a simple greeting?")

        print("SUCCESS: Gemini API test successful!")
        print(f"Response: {response.text}")
        return True

    except Exception as e:
        print(f"FAILED: Gemini API test failed: {e}")
        return False

if __name__ == "__main__":
    test_gemini_api()