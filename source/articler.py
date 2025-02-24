import os
import json
from typing import List, Dict
from glob import glob

def split_text_into_paragraphs(text: str, min_length: int = 50, split_on_single_newline: bool = True) -> List[str]:
    """Splits text into paragraphs based on newlines and filters by minimum length."""
    separator = '\n' if split_on_single_newline else '\n\n'
    paragraphs = [p.strip() for p in text.split(separator)]
    return [p for p in paragraphs if len(p) >= min_length]

def create_article_format(title: str, text: str, split_on_single_newline: bool = True) -> Dict:
    paragraphs = split_text_into_paragraphs(text, split_on_single_newline=split_on_single_newline)
    return {
        'title': title,
        'paragraphs': paragraphs
    }

def process_text_file(file_path: str, title: str = None, split_on_single_newline: bool = True) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if title is None:
        title = os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ').title()

    return create_article_format(title, text, split_on_single_newline)

def process_json_file(file_path: str, split_on_single_newline: bool = True) -> List[Dict]:
    """Process a JSON file containing article data.

    The JSON can either be a single article object or an array of article objects.
    All articles in the array will be processed.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert single object to list for consistent processing
    if not isinstance(data, list):
        data = [data]

    articles = []
    for article in data:
        title = article.get('title', '')
        content = article.get('content', '')
        articles.append(create_article_format(title, content, split_on_single_newline))

    return articles

def process_directory(directory: str, pattern: str = "*.*") -> List[Dict]:
    files = glob(os.path.join(directory, pattern))

    if not files:
        print(f"No files found matching pattern '{pattern}' in {directory}")
        return []

    articles = []
    for file_path in files:
        try:
            if file_path.lower().endswith('.json'):
                articles.extend(process_json_file(file_path))
            elif file_path.lower().endswith('.txt'):
                articles.append(process_text_file(file_path))
            else:
                continue

            print(f"Processed: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    return articles
