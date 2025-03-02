import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np

from .llm import LlmGenerator
from .color import print_colored

MAX_SUMMARY_LENGTH = 30000

class AdvancedRAG:
    def __init__(self, model_path):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') # Bi-encoder for retrieval
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') # Cross-encoder for re-ranking
        self.index = None
        self.chunks = []
        self.metadata = []
        self.llm = LlmGenerator(model_path)
        self.llm.load_model()

    def process_articles(self, articles):
        chunk_size = 5
        chunk_overlap = 1

        for article in articles:
            for para in article['paragraphs']:
                sentences = sent_tokenize(para)

                for i in range(0, len(sentences), chunk_size - chunk_overlap):
                    chunk = ' '.join(sentences[i:i+chunk_size])
                    self.chunks.append(chunk)
                    self.metadata.append({
                        'title': article['title'],
                        'para_index': len(self.metadata),
                        'start_sentence': i,
                        'end_sentence': min(i+chunk_size, len(sentences))
                    })

        # FAISS index
        embeddings = self.embedder.encode(self.chunks, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query, top_k=5):
        # First retrieve top-k results with bi-encoder
        query_embedding = self.embedder.encode([query])
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k*3)

        candidates = [self.chunks[i] for i in indices[0]]

        # Rerank top-k results with cross-encoder
        pairs = [(query, candidate) for candidate in candidates]
        rerank_scores = self.reranker.predict(pairs)

        combined = sorted(zip(indices[0], rerank_scores),
                         key=lambda x: x[1], reverse=True)[:top_k]

        return [{
            'text': self.chunks[idx],
            'score': score,
            'metadata': self.metadata[idx]
        } for idx, score in combined]

    def generate_answer(self, query, context_chunks):
        context = "\n\n".join([f"Source {i+1}: {chunk['text']}"
                             for i, chunk in enumerate(context_chunks)])

        print("\n\n")
        print_colored("Context:", 'title')
        print_colored(context, 'context')
        print("\n\n")

        prompt = f"""Instruction: Answer the question using only the provided context.

        Context:
        {context}

        Question: {query}

        Provide a concise bullet-point summary, followed by a brief paragraph elaborating on the key points. Don't mention sources. If answer is not in the sources, say 'I couldn't find that information'."""

        return self.llm.generate(prompt)


    def _truncate_for_summary(self, text: str) -> str:
        if len(text) > MAX_SUMMARY_LENGTH:
            print_colored("Text too long for summarization, truncating...", 'response')
            return text[:MAX_SUMMARY_LENGTH] + "..."
        return text

    def summarize_all_sources(self) -> dict:
        print_colored("\nGenerating summaries for each article...", 'title')

        article_summaries = {}
        unique_titles = set(meta['title'] for meta in self.metadata)

        for title in unique_titles:
            article_chunks = [
                chunk for chunk, meta in zip(self.chunks, self.metadata)
                if meta['title'] == title
            ]
            article_text = "\n\n".join(article_chunks)
            article_text = self._truncate_for_summary(article_text)

            prompt = f"""Instruction: Provide a 2-5 sentence summary of the following text:

            Text:
            {article_text}

            Create a very short and concise bullet-point summary that captures:
            - Main topics and key points
            - Important details and facts
            - Any significant conclusions or findings
            Respond only in plain text, without any formatting or special characters."""

            print_colored(f"\nSummarizing article: {title}", 'title')
            summary = self.llm.generate(prompt)
            article_summaries[title] = summary

        return article_summaries

    def summarize_specific_source(self, title: str) -> str:
        article_chunks = [
            chunk for chunk, meta in zip(self.chunks, self.metadata)
            if meta['title'].lower() == title.lower()
        ]

        if not article_chunks:
            return f"No article found with title: {title}"

        article_text = "\n\n".join(article_chunks)
        article_text = self._truncate_for_summary(article_text)

        print_colored(f"\nSummarizing article: {title}", 'title')

        prompt = f"""Instruction: Provide a very short summary of the following text:

        Text:
        {article_text}

        Create a very short and concise bullet-point summary that captures:
        - Main topics and key points
        - Important details and facts
        - Any significant conclusions or findings
        Respond only in plain text, without any formatting or special characters."""

        return self.llm.generate(prompt)

    def get_article_by_index_or_title(self, identifier: str) -> str:
        """Get article title by index or title string."""
        unique_titles = []
        seen = set()
        for meta in self.metadata:
            if meta['title'] not in seen:
                seen.add(meta['title'])
                unique_titles.append(meta['title'])

        try:
            if identifier.isdigit():
                index = int(identifier) - 1
                if 0 <= index < len(unique_titles):
                    return unique_titles[index]
        except ValueError:
            pass

        return identifier


    def __del__(self):
        if hasattr(self, 'llm'):
            self.llm.unload_model()
