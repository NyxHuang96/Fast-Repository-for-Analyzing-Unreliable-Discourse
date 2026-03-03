"""
File: website_scrape.py
Author: Darwin Zhang
Date: 2026-02-26
Course: COLX 523
Description: Scrape static HTM (mainly Reddit) and avoid ads along with random delays 
"""

import os
import time
import re
import requests
import pytesseract
from PIL import Image
from io import BytesIO
from typing import List, Dict
import jsonlines

class RedditScraper:
    """
    Extracts Title, Body, and runs OCR on Image URLs directly from Reddit's JSON API.
    """
    def __init__(self, subreddit: str):
        self.base_url = f"https://www.reddit.com/r/{subreddit}.json"
        # Custom User-Agent to avoid immediate 429 Too Many Requests errors
        # Following along Reddit's protocols, adding `self.header`
        self.headers = {
            'User-Agent': 'Python:FraudDataPipeline:v1.0 (by contact: p0g3b@cs.ubc.ca)'
        }

    def fetch_feed(self, limit: int = 25) -> List[Dict]:
        """Fetches the raw JSON feed from Reddit."""
        url = f"{self.base_url}?limit={limit}"
        print(f"[*] Hitting API: {url}")
        
        try:
            # Avoid too many requests and throttling 
            time.sleep(2) 
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return [child['data'] for child in response.json()['data']['children']]
        except requests.exceptions.RequestException as e:
            print(f"[!] API Request failed: {e}")
            return []

    def perform_ocr(self, image_url: str) -> str:
        """Downloads an image into memory and extracts text using Tesseract."""
        try:
            print(f"    [>] Running OCR on: {image_url}")
            response = requests.get(image_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            text = pytesseract.image_to_string(img)

            return text.strip()
        
        except Exception as e:
            print(f"    [!] OCR failed for {image_url}: {e}")
            return ""

    def process_posts(self, posts: List[Dict]) -> List[Dict]:
        """Maps Reddit data to our schema, running OCR if an image is detected."""
        dataset = []
        
        for post in posts:
            # Skip if ads 
            if post.get('promoted'):
                continue
                
            record = {
                # Record ID and meta data from post 
                "id": post.get('name'),
                "url": f"https://reddit.com{post.get('permalink')}",
                "victim_title": post.get('title', ''),
                "victim_body": post.get('selftext', ''),
                "ocr_text": "",
                "has_image": False
            }
            
            # If has image URL
            post_url = post.get('url', '')
            if post_url.endswith(('.jpg', '.png', '.jpeg')):
                record['has_image'] = True
                record['ocr_text'] = self.perform_ocr(post_url)
                
            # Keep the record if it has any viable text (Body or OCR)
            if record['victim_body'] or record['ocr_text']:
                dataset.append(record)
                
        return dataset

class NoiseFilter:
    """Filters out Reddit meta-posts, cries for help, and warnings."""
    
    def __init__(self):
        # Set regex patterns if it is a question (will add more later) instead of a spam/phishing text 
        self.meta_patterns = [
            re.compile(r"(?i)what do i do\?*"),
            re.compile(r"(?i)is this a scam\?*"),
            re.compile(r"(?i)am i safe\?*"),
            re.compile(r"(?i)help( me)? please"),
            re.compile(r"(?i)clicked (on )?a link"),
            re.compile(r"(?i)should i be worried\?*")
        ]

    def is_meta_post(self, text: str) -> bool:
        """Returns True if the text is likely commentary rather than a scam."""
        if not text:
            return False
            
        for pattern in self.meta_patterns:
            if pattern.search(text):
                return True
        return False

    def clean_dataset(self, records: List[Dict]) -> List[Dict]:
        """Filters the in-memory dataset to remove meta-noise."""
        clean_records = []
        dropped_count = 0
        
        for obj in records:
            title = obj.get("victim_title", "")
            body = obj.get("victim_body", "")
            
            # Try to drop if body is not related towards spam (meta-patterns)
            if self.is_meta_post(title) or self.is_meta_post(body[:200]):
                dropped_count += 1
                continue
            
            clean_records.append(obj)
            
        print(f"[*] Filtering complete.")
        print(f"    - Clean records retained: {len(clean_records)}")
        print(f"    - Meta-noise dropped: {dropped_count}")
        return clean_records

def save_to_jsonl(data: List[Dict], filepath: str):
    """Saves the extracted data to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with jsonlines.open(filepath, mode='w') as writer:
        for item in data:
            writer.write(item)
    print(f"[*] Saved {len(data)} records to {filepath}")


if __name__ == "__main__":
    # Look towards doing this on: r/phishing or r/Scams
    scraper = RedditScraper(subreddit="phishing") 
    noise_filter = NoiseFilter()
    
    # Fetch from API, and set limits 
    raw_posts = scraper.fetch_feed(limit=50) 
    print(f"[*] Found {len(raw_posts)} raw posts.")
    
    # Data and OCR 
    print("[*] Processing posts and running OCR...")
    structured_data = scraper.process_posts(raw_posts)
    
    # Filter out the meta-noise
    print("[*] Running heuristic noise filter...")
    clean_data = noise_filter.clean_dataset(structured_data)
    
    # Save the final output
    output_path = "data/raw/reddit_processed.jsonl"
    save_to_jsonl(clean_data, output_path)
    print("[*] Pipeline execution finished.")