import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
import re
import matplotlib.pyplot as plt
from nltk import download
from nltk.corpus import stopwords
from rake_nltk import Rake
from textblob import TextBlob
import os

# Download necessary NLTK resources 
download('stopwords')
download('punkt')

class BasicRealEstateScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
        }
        self.base_url = 'https://www.magicbricks.com'
        self.stop_words = set(stopwords.words('english'))
        self.rake = Rake(stopwords=self.stop_words)
        self.output_dir = 'output'  # Directory to save output files
        os.makedirs(self.output_dir, exist_ok=True)  # Create directory if it doesn't exist

    def extract_price(self, price_text):
        """Extract numeric price from text"""
        if not price_text:
            return None
        price_match = re.search(r'[\d.]+', price_text)
        if price_match:
            return float(price_match.group())
        return None

    def analyze_sentiment(self, text):
        """Analyze sentiment of a given text"""
        if text:
            blob = TextBlob(text)
            return blob.sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)
        return None

    def scrape_listings(self, city='Mumbai', max_pages=2):
        """Scrape basic property listings"""
        listings = []

        for page in range(1, max_pages + 1):
            try:
                url = f"{self.base_url}/property-for-sale/residential-real-estate?&cityName={city}&page={page}"
                print(f"Scraping page {page}: {url}")

                response = requests.get(url, headers=self.headers)
                if response.status_code != 200:
                    print(f"Failed to fetch page {page}. Status code: {response.status_code}")
                    continue

                soup = BeautifulSoup(response.content, 'html.parser')
                property_cards = soup.find_all('div', {'class': 'mb-srp__card'})
                print(f"Found {len(property_cards)} properties on page {page}")

                for card in property_cards:
                    try:
                        price_elem = card.find('div', {'class': 'mb-srp__card__price'})
                        title_elem = card.find('h2', {'class': 'mb-srp__card--title'})
                        area_elem = card.find('div', {'class': 'mb-srp__card__area'})

                        title = title_elem.text.strip() if title_elem else None
                        listing = {
                            'price': self.extract_price(price_elem.text.strip() if price_elem else None),
                            'title': title,
                            'area': area_elem.text.strip() if area_elem else None,
                            'url': self.base_url + card.find('a')['href'] if card.find('a') else None,
                            'sentiment': self.analyze_sentiment(title)
                        }

                        if listing['price']:
                            listings.append(listing)
                            print(f"Scraped listing: {listing['title']} - {listing['price']} - Sentiment: {listing['sentiment']}")

                    except Exception as e:
                        print(f"Error processing property card: {str(e)}")
                        continue

                sleep(2)

            except Exception as e:
                print(f"Error scraping page {page}: {str(e)}")
                continue

        df = pd.DataFrame(listings)
        print(f"\nTotal listings scraped: {len(df)}")
        return df

    def analyze_keywords(self, df):
        """Extract keywords from listing titles using RAKE"""
        all_titles = " ".join(df['title'].dropna())
        self.rake.extract_keywords_from_text(all_titles)
        keywords = self.rake.get_ranked_phrases_with_scores()

        keywords_df = pd.DataFrame(keywords, columns=['score', 'keyword'])
        keywords_df = keywords_df.drop_duplicates().sort_values(by='score', ascending=False).reset_index(drop=True)  # Remove duplicates and sort
        top_keywords = keywords_df.head(20)  # Top 20 keywords by score
        print("\nTop Keywords:\n", top_keywords)
        return top_keywords

    def visualize_data(self, df, keywords_df):
        """Visualize the price distribution and top keywords"""
        # Plot price distribution
        plt.figure(figsize=(10, 5))
        plt.hist(df['price'].dropna(), bins=20, color='skyblue', edgecolor='black')
        plt.title('Price Distribution of Properties')
        plt.xlabel('Price (in lakhs or crores)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'price_distribution.png'))  # Save figure
        plt.show()

        # Plot top keywords
        plt.figure(figsize=(10, 5))
        plt.barh(keywords_df['keyword'], keywords_df['score'], color='salmon')
        plt.xlabel('Keyword Score')
        plt.title('Top Keywords in Property Listings')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'top_keywords.png'))  # Save figure
        plt.show()

        # Plot sentiment distribution
        plt.figure(figsize=(10, 5))
        plt.hist(df['sentiment'].dropna(), bins=20, color='lightgreen', edgecolor='black')
        plt.title('Sentiment Analysis of Listing Titles')
        plt.xlabel('Sentiment Polarity')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sentiment_distribution.png'))  # Save figure
        plt.show()

def main():
    scraper = BasicRealEstateScraper()
    print("Starting to scrape listings...")
    df = scraper.scrape_listings(city='Mumbai', max_pages=2)
    csv_filename = os.path.join(scraper.output_dir, 'raw_listings.csv')  # Path to save CSV
    df.to_csv(csv_filename, index=False)
    print(f"\nRaw data saved to '{csv_filename}'")

    print("\nData Summary:")
    print(df.info())
    print("\nSample of scraped data:")
    print(df.head())

    # Semantic analysis - keyword extraction
    keywords_df = scraper.analyze_keywords(df)

    # Visualize results
    scraper.visualize_data(df, keywords_df)

if __name__ == "__main__":
    main()
