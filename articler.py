import os
from typing import List, Dict
from glob import glob

def split_text_into_paragraphs(text: str, min_length: int = 20) -> List[str]:
    """Splits text into paragraphs based on double newlines and filters by minimum length."""
    paragraphs = [p.strip() for p in text.split('\n\n')]
    return [p for p in paragraphs if len(p) >= min_length]

def create_article_format(title: str, text: str) -> Dict:
    paragraphs = split_text_into_paragraphs(text)
    return {
        'title': title,
        'paragraphs': paragraphs
    }

def process_text_file(file_path: str, title: str = None) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if title is None:
        title = os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ').title()

    return create_article_format(title, text)

def process_directory(directory: str, pattern: str = "*.txt") -> List[Dict]:
    files = glob(os.path.join(directory, pattern))

    if not files:
        print(f"No files found matching pattern '{pattern}' in {directory}")
        return []

    file_path = files[0]
    try:
        article = process_text_file(file_path)
        print(f"Processed: {file_path}")
        return [article]
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []
