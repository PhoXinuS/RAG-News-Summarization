import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np

from llm import LlmGenerator
from color import print_colored
from articler import process_json_file, process_text_file

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


if __name__ == "__main__":
    model_path = r"X:\Path\To\HuggingFace\Model"

    articles = [
        {
            'title': 'Recipe for coffee jelly dessert',
            'paragraphs': [
                'Coffee jelly (コーヒーゼリー, kōhī zerī) is a jelly dessert flavored with coffee and sugar.[1][2] Although once common in British and American cookbooks, it is now most common in Japan, where it can be found in most restaurants and convenience stores. Coffee jelly can be made using instant mix or from scratch. It is served in restaurants and cafés. Coffee jelly is also frequently used in bubble tea/coffees.',
                'In the early 20th century coffee jelly was promoted as a healthier alternative to hot coffee, as it was thought the gelatin would absorb excess acid in the stomach.[7]',
                'Jell-O launched a short lived coffee gelatin mix in 1918,[8] but the dessert never gained widespread popularity outside of New England. Today, coffee jelly may still be found in Rhode Island, Massachusetts and other New England states. Durgin-Park restaurant in Boston, which opened in 1827, still offered coffee gelatin made with leftover coffee from the previous day as of 2016.[9]',
                'Japanese coffee jelly was developed during the Taishō period (1912–1926)[10] in imitation of European molded jellies. It appealed to modern young men with tastes for Western fashion and rose in popularity along with café culture.[10] Coffee jelly has remained popular in Japan and is still widely available. Starbucks launched a coffee jelly frappuccino in Japan in 2016.[11][12]',
                'Description: Coffee jelly is made from sweetened coffee added to agar, a gelatinous substance made from algae and called kanten in Japanese.[10] It may also be made from gelatin rather than agar, which is more common in European and American cuisine.',
                'It is often cut into cubes and served in a variety of dessert dishes and beverages. Cubes of coffee jelly are sometimes added to milkshakes, at the bottom of an ice cream float, or to garnish an ice cream sundae. Coffee jelly is often added to a cup of hot or iced coffee, with cream and gum syrup added. Condensed milk is poured over cubes of chilled coffee jelly in a bowl.[13]',
                'Popular Culture: Coffee jelly has appeared numerous times in the manga and anime series The Disastrous Life of Saiki K., in which it is the main character Saiki Ks favorite food.'
            ]
        }
    ]

    overwatch_text = process_text_file(r".\Overwatch_Article.txt")
    articles.append(overwatch_text)

    tanenbaum_text = process_text_file(r".\Minix_Bible.txt")
    articles.append(tanenbaum_text)

    geekwire = process_json_file(r".\scraped_articles\articles_20250219_124453.json")
    articles.extend(geekwire)

    try:
        rag = AdvancedRAG(model_path)
        print_colored("Processing articles...", 'title')
        rag.process_articles(articles)

        while True:
            try:
                print_colored("\n=== RAG Query Interface ===\n", 'title')
                print_colored("Available commands:", 'title')
                print_colored("  list, l               - Show available articles", 'response')
                print_colored("  summarize             - Generate summary of all articles", 'response')
                print_colored("  summarize <num/title> - Generate summary of specific article", 'response')
                print_colored("  quit, q               - Exit the program", 'response')
                print_colored("\nEnter your question or command: ", 'title')
                query = input().strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    break

                if any(query.lower().startswith(cmd) for cmd in ['summarize ', 'sum ', 's ']):
                    if query.lower().startswith('summarize '):
                        identifier = query[9:].strip()
                    elif query.lower().startswith('sum '):
                        identifier = query[4:].strip()
                    else:  # s command
                        identifier = query[2:].strip()

                    title = rag.get_article_by_index_or_title(identifier)
                    print_colored(f"\nGenerating summary for article: {title}", 'title')
                    summary = rag.summarize_specific_source(title)
                    print_colored("\nSummary:", 'title')
                    print_colored(summary, 'response')
                    continue
                elif query.lower() in ['summarize', 'summary', 's', 'sum']:
                    print_colored("\nGenerating comprehensive summary of all sources...", 'title')
                    summaries = rag.summarize_all_sources()
                    print_colored("\nIndividual Article Summaries:", 'title')
                    for title, summary in summaries.items():
                        print_colored(f"\n\n{title}:", 'title')
                        print_colored(summary, 'response')
                    continue

                if query.lower() in ['list', 'l']:
                    print_colored("\n=== Available Articles ===\n", 'title')
                    unique_titles = []
                    seen = set()
                    for meta in rag.metadata:
                        if meta['title'] not in seen:
                            seen.add(meta['title'])
                            unique_titles.append(meta['title'])

                    max_index_width = len(str(len(unique_titles)))
                    for i, title in enumerate(unique_titles, 1):
                        index = str(i).rjust(max_index_width)
                        print_colored(f"  [{index}] {title}", 'response')
                    print()
                    continue

                if not query:
                    print_colored("Please enter a valid question!", 'error')
                    continue

                print_colored("\nSearching relevant context...", 'title')
                results = rag.search(query)

                print_colored("\nGenerating answer...", 'title')
                answer = rag.generate_answer(query, results)

                print_colored("\nAnswer:", 'title')
                print_colored(answer, 'response')

            except KeyboardInterrupt:
                print_colored("\nInterrupt received, exiting...", 'error')
                break
            except Exception as e:
                print_colored(f"\nError: {str(e)}", 'error')
                continue
    finally:
        del rag
        print_colored("\nExiting...", 'blue')
