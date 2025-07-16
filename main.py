import os
from google_play_scraper import reviews_all, Sort
import csv
from datetime import datetime

class GooglePlayReviewScraper:
    def __init__(self, app_id, app_name, score=1, lang='id', country='id', sleep_milliseconds=0):
        self.app_id = app_id
        self.app_name = app_name
        self.score = score
        self.lang = lang
        self.country = country
        self.sleep_milliseconds = sleep_milliseconds
        self.reviews = []
        self.output_dir = "reviews"
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def fetch_reviews(self):
        print(f"Fetching {self.score}-star reviews for {self.app_name}...")
        try:
            self.reviews = reviews_all(
                self.app_id,
                sleep_milliseconds=self.sleep_milliseconds,
                lang=self.lang,
                country=self.country,
                filter_score_with=self.score,
                sort=Sort.MOST_RELEVANT,
            )
            print(f"Total reviews fetched for {self.app_name}: {len(self.reviews)}")
        except Exception as e:
            print(f"Error fetching reviews for {self.app_name}: {e}")
        return self.reviews

    def save_reviews_to_file(self):
        filename_txt = f"{self.output_dir}/reviews_{self.score}_star_{self.app_name.lower().replace(' ', '_')}.txt"
        filename_csv = filename_txt.replace(".txt", ".csv")

        with open(filename_txt, 'w', encoding='utf-8') as file:
            for review in self.reviews:
                date_str = review['at'].strftime("%Y-%m-%d %H:%M:%S") if review.get('at') else "N/A"
                file.write(f"Score: {review['score']} User: {review['userName']} Message: {review['content']} Date: {date_str}\n")
        print(f"Saved to: {filename_txt}")

        with open(filename_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['score', 'user', 'message', 'date'])
            for review in self.reviews:
                date_str = review['at'].strftime("%Y-%m-%d %H:%M:%S") if review.get('at') else "N/A"
                message = review.get('content')
                if message is None:
                    message = ''
                else:
                    message = message.replace('\n', ' ').strip()

                writer.writerow([
                    review.get('score', ''),
                    review.get('userName', ''),
                    message,
                    date_str
                ])
        print(f"Saved to: {filename_csv}")


    def run(self):
        self.fetch_reviews()
        self.save_reviews_to_file()

if __name__ == "__main__":
    apps = [
        {"id": "com.telkomsel.telkomselcm", "name": "MyTelkomsel"},
        # {"id": "com.shopee.id", "name": "Shopee"},
        # {"id": "com.tokopedia.tkpd", "name": "Tokopedia"},
        # {"id": "com.ss.android.ugc.trill", "name": "TikTok"},
        # {"id": "id.bmri.livin", "name": "Livin Mandiri"},
        # {"id": "com.pure.indosat.care", "name": "MyIM3"},
    ]

    for app in apps:
        for star in range(1, 6):
            scraper = GooglePlayReviewScraper(
                app_id=app["id"],
                app_name=app["name"],
                score=star
            )
            scraper.run()