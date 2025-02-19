from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from typing import List, Optional, Dict
from selenium.webdriver import Chrome
from datetime import datetime, date
import json
import time
import os


def _generate_target_years(start_date: date, end_date: date) -> List[str]:
    """Generates list of years in range for URL validation"""
    return [str(year) for year in range(start_date.year, end_date.year + 1)]


def _is_valid_year(url: str, target_years: List[str]) -> bool:
    """Validates if URL contains any of target years"""
    return any(f"/{year}/" in url for year in target_years)


def _parse_article_date(date_str: str) -> Optional[date]:
    """Safely parses date from string"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").date()
    except (ValueError, TypeError):
        return None


def _clean_article_content(soup: BeautifulSoup) -> str:
    """Cleans and extracts text from article content"""
    content_div = soup.find("div", class_="entry-content clearfix")
    if not content_div:
        return "Article content not found."

    for element in content_div.find_all(["script", "style", "iframe",
                                         "figure", "div", "footer"]):
        element.decompose()

    return content_div.get_text().strip()


def _process_table_row(row, target_years: List[str],
                       start_date: date, end_date: date) -> Optional[Dict]:
    """Processes individual table row from sitemap"""
    columns = row.find_all("td")
    if len(columns) < 3:
        return None

    try:
        url = columns[1].find("a")["href"]
        lastmod_text = columns[2].text.strip()
    except (AttributeError, KeyError):
        return None

    if not _is_valid_year(url, target_years):
        return None

    article_date = _parse_article_date(lastmod_text)
    if not article_date:
        return None

    if start_date <= article_date <= end_date:
        return {
            "date": article_date.isoformat(),
            "url": url
        }
    return None


class Scraper:
    def __init__(self):
        self._init_driver()
        self.output_dir = "scraped_articles"
        os.makedirs(self.output_dir, exist_ok=True)

    def _init_driver(self):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        self.driver = Chrome(options=options)

    def _close_driver(self):
        self.driver.quit()

    def scrape(self, start_date: date, end_date: Optional[date] = None) -> Optional[str]:
        """Main method to initiate scraping process"""
        sitemap_url = "https://www.geekwire.com/sitemap-27.xml"
        print("Fetching articles...")
        articles = self._get_article_links(sitemap_url, start_date, end_date)

        if not articles:
            print("No articles found in specified date range.")
            return None

        print(f"Found {len(articles)} articles. Fetching content...")
        results = []
        for i, article in enumerate(articles, 1):
            content_data = self._get_article_content(article["url"])
            results.append({
                **article,
                **content_data
            })
            print(f"Fetched {i}/{len(articles)} articles.")

        print("All articles saved successfully.")
        return self._save_results(results)

    def _get_article_links(self, sitemap_url: str, start_date: date,
                           end_date: Optional[date] = None) -> List[Dict]:
        """Retrieves article links within specified date range"""
        end_date = end_date or start_date
        start_date, end_date = sorted([start_date, end_date])

        target_years = _generate_target_years(start_date, end_date)

        try:
            self.driver.get(sitemap_url)
            time.sleep(5)
            return self._parse_sitemap_content(target_years, start_date, end_date)
        finally:
            self._close_driver()

    def _parse_sitemap_content(self, target_years: List[str],
                               start_date: date, end_date: date) -> List[Dict]:
        """Parses sitemap content to extract relevant links"""
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        articles = []

        for row in soup.select("table tbody tr"):
            if article := _process_table_row(row, target_years, start_date, end_date):
                articles.append(article)
        return articles

    def _get_article_content(self, article_url: str) -> Dict:
        """Extracts title and content from article page"""
        try:
            self._init_driver()
            self.driver.get(article_url)
            time.sleep(3)
            soup = BeautifulSoup(self.driver.page_source, "html.parser")

            # Extract title from h1
            title_tag = soup.find("h1")
            title = title_tag.text.strip() if title_tag else "No title found"

            # Extract content
            content = _clean_article_content(soup)

            return {
                "title": title,
                "content": content
            }
        except Exception as e:
            print(f"Error fetching content from {article_url}: {str(e)}")
            return {
                "title": "Error",
                "content": "Content unavailable"
            }
        finally:
            self._close_driver()

    def _save_results(self, data: List[Dict]) -> str:
        """Saves results to JSON file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"articles_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return filepath


if __name__ == "__main__":
    scraper = Scraper()
    result_file = scraper.scrape(date(2025, 2, 16), date(2025, 2, 18))
    if result_file:
        print(f"Results saved to: {result_file}")