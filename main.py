from source.rag import AdvancedRAG
from source.scraper import Scraper
from source.articler import process_text_file, process_json_file
from source.color import print_colored
from datetime import date, timedelta
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    model_path = os.getenv("MODEL_PATH")

    articles = []
    overwatch_text = process_text_file(r".\data\Overwatch_Article.txt")
    articles.append(overwatch_text)

    tanenbaum_text = process_text_file(r".\data\Minix_Bible.txt")
    articles.append(tanenbaum_text)

    scraper = Scraper()
    yesterday = date.today() - timedelta(days=1)
    today = date.today()
    print_colored(f"Fetching articles from {yesterday} to {today}", 'context')

    scraped_articles_output = scraper.scrape(yesterday, today)
    process_geekwire_articles = process_json_file(scraped_articles_output)
    articles.extend(process_geekwire_articles)

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
                    else:
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
        if rag is not None:
            del rag
        print_colored("\nExiting...", 'blue')

if __name__ == "__main__":
    main()