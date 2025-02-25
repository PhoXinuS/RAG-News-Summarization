from rag import AdvancedRAG
from scraper import Scraper
from articler import process_text_file, process_json_file
from color import print_colored
from datetime import date, datetime, timedelta

def main():
    model_path = r"X:\llm-fun\pip-test\ConvertedLlama1BInstruct"

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

    scraper = Scraper()
    yesterday = date.today() - timedelta(days=1)
    scraped_articles_output = scraper.scrape(yesterday)
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