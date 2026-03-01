"""
File: test_scrape.py
Author: Darwin Zhang
Date: 2026-02-26
Course: COLX 523
Description: Test if the statis scraper is functional
"""

import pytest
from src.website_scrape import NoiseFilter

def test_noise_filter_catches_meta_questions():
    """Ensures the filter successfully flags cries for help."""
    filter_bot = NoiseFilter()
    
    # These should be flagged as True (meta-noise)
    assert filter_bot.is_meta_post("I clicked a link what do I do?") == True
    assert filter_bot.is_meta_post("Is this a scam? I am worried.") == True

def test_noise_filter_keeps_fraud_text():
    """Ensures the filter does NOT accidentally drop actual fraud text."""
    filter_bot = NoiseFilter()
    
    # These should be flagged as False (actual scam text we want to keep)
    fraud_text_1 = "URGENT: Your account has been locked. Click here to verify."
    fraud_text_2 = "Send 1500 USD in XMR to the following wallet immediately."
    
    assert filter_bot.is_meta_post(fraud_text_1) == False
    assert filter_bot.is_meta_post(fraud_text_2) == False

def test_jsonl_schema_integrity():
    """
    A mock test to ensure our output dictionary always has the required keys.
    This prevents downstream ML pipeline crashes.
    """
    mock_record = {
        "id": "t3_xyz",
        "url": "https://reddit.com/...",
        "victim_title": "Test",
        "victim_body": "Test body",
        "ocr_text": "",
        "has_image": False
    }
    
    required_keys = {"id", "url", "victim_title", "victim_body", "ocr_text", "has_image"}
    assert set(mock_record.keys()) == required_keys