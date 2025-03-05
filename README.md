# RAG News Summarization App

## Project Overview

The AI-powered tool, developed for the GOLEM Machine Learning Student Scientific Association, that uses **Retrieval-Augmented Generation (RAG)** to scrape and summarize the latest tech news from GeekWire.

## Technologies Used

-   **Language Models**: [Qwen 2.5 3B Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
-   **Vector Search**: FAISS (*Facebook AI Similarity Search*)
-   **Embeddings**: [Bi-Encoder](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [Cross-Encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) 
-   **Web Scraping**: Selenium, BeautifulSoup
-   **Natural Language Processing**: NLTK (*Natural Language Toolkit*)

## Installation and Usage

### Installation
**Make sure to use a virtual environment with Python 3.11.**
Clone the repository and install the required dependencies.


```bash
# Clone the repository
git clone https://github.com/USERNAME/rag-news-summarizer
cd rag-news-summarizer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies for RAG functionality
pip install -r requirements_rag.txt

# Install dependencies for web scraping
pip install -r requirements_scraper.txt
```
Create a `.env` file based on the `.env.example` template. It should store the path to your model.  
If you need to convert LLaMA weights to Hugging Face format, use the following command:

```bash
python convert_llama_weights_to_hf.py --input_dir /path/to/llama/weights --model_size 7B --output_dir ./models/llama-7b-hf
```

### Running the Project

To run the project with default settings (scraping latest articles):

```bash
python main.py
```

Once the application is running, you can use the following commands:

-   Type a natural language question to search across all articles
-   Use `list` or `l` to display all available articles
-   Use `summarize` to generate summaries of all articles
-   Use `summarize <number or title>` to summarize a specific article
-   Use `quit` or `q` to exit the application

Optionally, you can run the LLM module separately:
```bash
python source/llm.py
```
### Examples
#### Scraper
![Scraper](https://raw.githubusercontent.com/PhoXinuS/RAG-News-Summarization/refs/heads/main/presentation/image1.png)
#### Interface
![Interface](https://github.com/PhoXinuS/RAG-News-Summarization/blob/main/presentation/image3.png?raw=true)
#### Context
![Context](https://raw.githubusercontent.com/PhoXinuS/RAG-News-Summarization/refs/heads/main/presentation/image2.png)
#### Answer
![Answer](https://github.com/PhoXinuS/RAG-News-Summarization/blob/main/presentation/image4.png?raw=true)
