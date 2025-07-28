import os
import csv
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from transformers import pipeline
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from datetime import datetime
from google_play_scraper import reviews_all, Sort
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GooglePlayReviewScraper:
    def __init__(self, app_id: str, app_name: str, score: int = 1, lang: str = 'id', 
                 country: str = 'id', sleep_milliseconds: int = 0, max_reviews: Optional[int] = None):
        self.app_id = app_id
        self.app_name = app_name
        self.score = score
        self.lang = lang
        self.country = country
        self.sleep_milliseconds = sleep_milliseconds
        self.max_reviews = max_reviews
        self.reviews = []
        self.output_dir = "reviews"
        self._ensure_output_dir()
        self.logger = logging.getLogger(__name__)

    def _ensure_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f"Created output directory: {self.output_dir}")

    def fetch_reviews(self) -> List[Dict]:
        self.logger.info(f"Fetching {self.score}-star reviews for {self.app_name}...")
        try:
            # Fetch reviews with count limit if specified
            kwargs = {
                'sleep_milliseconds': self.sleep_milliseconds,
                'lang': self.lang,
                'country': self.country,
                'filter_score_with': self.score,
                'sort': Sort.MOST_RELEVANT,
            }
            
            if self.max_reviews:
                kwargs['count'] = self.max_reviews
                
            self.reviews = reviews_all(self.app_id, **kwargs)
            self.logger.info(f"Total reviews fetched for {self.app_name}: {len(self.reviews)}")
            
        except Exception as e:
            self.logger.error(f"Error fetching reviews for {self.app_name}: {e}")
            self.reviews = []
        return self.reviews

    def save_reviews_to_file(self):
        if not self.reviews:
            self.logger.warning("No reviews to save")
            return
            
        filename_base = f"{self.output_dir}/reviews_{self.score}_star_{self.app_name.lower().replace(' ', '_')}"
        filename_txt = f"{filename_base}.txt"
        filename_csv = f"{filename_base}.csv"
        filename_json = f"{filename_base}.json"

        # Save as TXT
        with open(filename_txt, 'w', encoding='utf-8') as file:
            for review in self.reviews:
                date_str = review['at'].strftime("%Y-%m-%d %H:%M:%S") if review.get('at') else "N/A"
                file.write(f"Score: {review['score']} User: {review['userName']} Message: {review['content']} Date: {date_str}\n")
        self.logger.info(f"Saved to: {filename_txt}")

        # Save as CSV
        with open(filename_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['score', 'user', 'message', 'date', 'thumbs_up', 'reply_content'])
            for review in self.reviews:
                date_str = review['at'].strftime("%Y-%m-%d %H:%M:%S") if review.get('at') else "N/A"
                message = review.get('content', '').replace('\n', ' ').strip() if review.get('content') else ''
                
                writer.writerow([
                    review.get('score', ''),
                    review.get('userName', ''),
                    message,
                    date_str,
                    review.get('thumbsUpCount', 0),
                    review.get('replyContent', '').replace('\n', ' ').strip() if review.get('replyContent') else ''
                ])
        self.logger.info(f"Saved to: {filename_csv}")

        # Save as JSON for full data preservation
        with open(filename_json, 'w', encoding='utf-8') as jsonfile:
            # Convert datetime objects to strings for JSON serialization
            reviews_json = []
            for review in self.reviews:
                review_copy = review.copy()
                if review_copy.get('at'):
                    review_copy['at'] = review_copy['at'].isoformat()
                reviews_json.append(review_copy)
            json.dump(reviews_json, jsonfile, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved to: {filename_json}")

    def get_review_statistics(self) -> Dict:
        if not self.reviews:
            return {}
            
        stats = {
            'total_reviews': len(self.reviews),
            'avg_score': sum(r.get('score', 0) for r in self.reviews) / len(self.reviews),
            'score_distribution': {},
            'avg_thumbs_up': sum(r.get('thumbsUpCount', 0) for r in self.reviews) / len(self.reviews),
            'reviews_with_reply': sum(1 for r in self.reviews if r.get('replyContent'))
        }
        
        # Score distribution
        for i in range(1, 6):
            stats['score_distribution'][i] = sum(1 for r in self.reviews if r.get('score') == i)
            
        return stats

    def run(self):
        self.fetch_reviews()
        if self.reviews:
            self.save_reviews_to_file()
            stats = self.get_review_statistics()
            self.logger.info(f"Review statistics: {stats}")
        return self.reviews

class ReviewProcessor:
    def __init__(self, reviews: List[Dict], output_dir: str = "reviews"):
        self.reviews = reviews
        self.output_dir = output_dir
        self.stopword_factory = StopWordRemoverFactory()
        self.stemmer_factory = StemmerFactory()
        
        # Fix for Sastrawi stopwords
        try:
            self.stopwords = set(self.stopword_factory.get_stop_words())  # Fixed method name
        except AttributeError:
            # Fallback to manual stopwords if library method doesn't work
            self.stopwords = {
                'yang', 'dan', 'di', 'ke', 'dari', 'dalam', 'untuk', 'pada', 'dengan', 'adalah', 
                'ini', 'itu', 'tidak', 'atau', 'juga', 'akan', 'dapat', 'ada', 'sudah', 'harus',
                'bisa', 'kalau', 'karena', 'jika', 'saya', 'kamu', 'dia', 'mereka', 'kita', 'kami',
                'nya', 'an', 'kan', 'lah', 'kah', 'pun', 'per', 'se', 'ter', 'ber', 'me', 'pen'
            }
        
        self.stemmer = self.stemmer_factory.create_stemmer()
        self.logger = logging.getLogger(__name__)
        
        # Additional Indonesian stopwords
        additional_stopwords = {
            'app', 'aplikasi', 'bagus', 'jelek', 'mantap', 'keren', 'buruk', 'baik', 'ok', 'oke',
            'sangat', 'sekali', 'banget', 'bgt', 'gak', 'ga', 'nggak', 'enggak', 'udah', 'udh',
            'sih', 'aja', 'deh', 'dong', 'kok', 'tapi', 'trus', 'terus', 'jadi', 'biar', 'buat',
            'kalo', 'kayak', 'gimana', 'kenapa', 'apa', 'siapa', 'dimana', 'kapan', 'mana'
        }
        self.stopwords.update(additional_stopwords)

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        if not text:
            return ""
            
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        words = [word for word in text.split() if word not in self.stopwords and len(word) > 2]
        
        # Stemming
        stemmed_text = self.stemmer.stem(' '.join(words))
        
        return stemmed_text

    def preprocess_reviews(self) -> List[str]:
        """Preprocess all review contents"""
        processed_reviews = []
        for review in self.reviews:
            content = review.get('content', '')
            processed_content = self.preprocess_text(content)
            if processed_content:  # Only add non-empty processed content
                processed_reviews.append(processed_content)
        
        self.logger.info(f"Processed {len(processed_reviews)} reviews out of {len(self.reviews)}")
        return processed_reviews
    
    def generate_wordcloud(self, processed_reviews: List[str], app_name: str = "app"):
        """Generate and save word cloud with frequency analysis"""
        if not processed_reviews:
            self.logger.warning("No processed reviews to generate wordcloud")
            return
            
        text = ' '.join(processed_reviews)
        if not text.strip():
            self.logger.warning("No meaningful text to generate wordcloud")
            return
        
        try:
            # Create word frequency analysis
            from collections import Counter
            words = text.split()
            word_freq = Counter(words)
            
            # Save word frequency data
            freq_data = [{'word': word, 'frequency': freq} for word, freq in word_freq.most_common(50)]
            freq_df = pd.DataFrame(freq_data)
            freq_output_path = f"{self.output_dir}/word_frequency_{app_name.lower().replace(' ', '_')}.csv"
            freq_df.to_csv(freq_output_path, index=False)
            self.logger.info(f"Word frequency data saved to {freq_output_path}")
            
            # Generate basic wordcloud
            wordcloud = WordCloud(
                width=1200, 
                height=600, 
                background_color='white',
                max_words=100,
                colormap='viridis',
                relative_scaling=0.5,
                min_font_size=10,
                prefer_horizontal=0.7
            ).generate(text)
            
            # Save basic wordcloud
            basic_output_path = f"{self.output_dir}/wordcloud_{app_name.lower().replace(' ', '_')}.png"
            wordcloud.to_file(basic_output_path)
            self.logger.info(f"Basic word cloud saved to {basic_output_path}")
            
            # Create enhanced wordcloud with matplotlib
            plt.figure(figsize=(16, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud - {app_name}', fontsize=20, fontweight='bold', pad=20)
            plt.tight_layout()
            enhanced_output_path = f"{self.output_dir}/wordcloud_enhanced_{app_name.lower().replace(' ', '_')}.png"
            plt.savefig(enhanced_output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            self.logger.info(f"Enhanced word cloud saved to {enhanced_output_path}")
            
            # Create frequency bar chart
            plt.figure(figsize=(12, 8))
            top_words = word_freq.most_common(20)
            words, frequencies = zip(*top_words)
            
            bars = plt.bar(range(len(words)), frequencies, color='skyblue', edgecolor='navy', alpha=0.7)
            plt.xlabel('Words', fontsize=12, fontweight='bold')
            plt.ylabel('Frequency', fontsize=12, fontweight='bold')
            plt.title(f'Top 20 Most Frequent Words - {app_name}', fontsize=14, fontweight='bold')
            plt.xticks(range(len(words)), words, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, freq in zip(bars, frequencies):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{freq}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            freq_chart_path = f"{self.output_dir}/word_frequency_chart_{app_name.lower().replace(' ', '_')}.png"
            plt.savefig(freq_chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Word frequency chart saved to {freq_chart_path}")
            
            # Create different colored wordclouds
            colormaps = ['plasma', 'inferno', 'magma', 'cividis']
            for i, colormap in enumerate(colormaps):
                colored_wordcloud = WordCloud(
                    width=1200, 
                    height=600, 
                    background_color='white',
                    max_words=100,
                    colormap=colormap,
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate(text)
                
                colored_output_path = f"{self.output_dir}/wordcloud_{colormap}_{app_name.lower().replace(' ', '_')}.png"
                colored_wordcloud.to_file(colored_output_path)
            
            self.logger.info(f"Multiple colored wordclouds generated for {app_name}")
            
            # Print top words summary
            print(f"\n📊 TOP 10 MOST FREQUENT WORDS - {app_name}")
            print("-" * 40)
            for i, (word, freq) in enumerate(word_freq.most_common(10), 1):
                print(f"{i:2d}. {word:<15} : {freq:>4} times")
            print("-" * 40)
            
        except Exception as e:
            self.logger.error(f"Error generating wordcloud: {e}")
            # Fallback to simple wordcloud
            try:
                simple_wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white'
                ).generate(text)
                
                fallback_path = f"{self.output_dir}/wordcloud_simple_{app_name.lower().replace(' ', '_')}.png"
                simple_wordcloud.to_file(fallback_path)
                self.logger.info(f"Fallback wordcloud saved to {fallback_path}")
            except Exception as fallback_error:
                self.logger.error(f"Fallback wordcloud also failed: {fallback_error}")
    
    def sentiment_analysis(self, app_name: str = "app"):
        """Enhanced sentiment analysis"""
        if not self.reviews:
            self.logger.warning("No reviews for sentiment analysis")
            return
            
        try:
            # Use a more robust sentiment model
            sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                return_all_scores=True
            )
            
            contents = [review.get('content', '') for review in self.reviews if review.get('content')]
            if not contents:
                self.logger.warning("No review contents for sentiment analysis")
                return
                
            # Process in batches to avoid memory issues
            batch_size = 50
            all_sentiments = []
            
            for i in range(0, len(contents), batch_size):
                batch = contents[i:i+batch_size]
                batch_sentiments = sentiment_pipeline(batch)
                all_sentiments.extend(batch_sentiments)
            
            # Process results
            sentiment_data = []
            for i, sentiment_scores in enumerate(all_sentiments):
                best_sentiment = max(sentiment_scores, key=lambda x: x['score'])
                sentiment_data.append({
                    'review_index': i,
                    'content': contents[i][:100] + '...' if len(contents[i]) > 100 else contents[i],
                    'sentiment': best_sentiment['label'],
                    'confidence': best_sentiment['score'],
                    'score': self.reviews[i].get('score', 0)
                })
            
            # Create DataFrame and save
            df = pd.DataFrame(sentiment_data)
            output_path = f"{self.output_dir}/sentiment_analysis_{app_name.lower().replace(' ', '_')}.csv"
            df.to_csv(output_path, index=False)
            
            # Generate sentiment summary
            sentiment_summary = df['sentiment'].value_counts()
            avg_confidence = df['confidence'].mean()
            
            summary = {
                'total_reviews': len(df),
                'sentiment_distribution': sentiment_summary.to_dict(),
                'average_confidence': avg_confidence,
                'positive_percentage': (sentiment_summary.get('LABEL_2', 0) / len(df)) * 100,
                'negative_percentage': (sentiment_summary.get('LABEL_0', 0) / len(df)) * 100,
                'neutral_percentage': (sentiment_summary.get('LABEL_1', 0) / len(df)) * 100
            }
            
            # Save summary
            with open(f"{self.output_dir}/sentiment_summary_{app_name.lower().replace(' ', '_')}.json", 'w') as f:
                json.dump(summary, f, indent=2)
                
            self.logger.info(f"Sentiment analysis completed and saved to {output_path}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return None
    
    def topic_modeling(self, processed_reviews: List[str], app_name: str = "app", n_topics: int = 5):
        """Perform topic modeling using LDA"""
        if len(processed_reviews) < n_topics:
            self.logger.warning(f"Not enough reviews ({len(processed_reviews)}) for {n_topics} topics")
            return
            
        try:
            # Vectorize the text
            vectorizer = TfidfVectorizer(
                max_features=100,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2)
            )
            
            doc_term_matrix = vectorizer.fit_transform(processed_reviews)
            
            # LDA Topic Modeling
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            lda.fit(doc_term_matrix)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    'topic_id': topic_idx,
                    'top_words': top_words,
                    'weights': [topic[i] for i in top_words_idx]
                })
            
            # Save topics
            output_path = f"{self.output_dir}/topics_{app_name.lower().replace(' ', '_')}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(topics, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Topic modeling completed and saved to {output_path}")
            return topics
            
        except Exception as e:
            self.logger.error(f"Error in topic modeling: {e}")
            return None

    def create_visualizations(self, app_name: str = "app"):
        """Create various visualizations"""
        if not self.reviews:
            return
            
        try:
            # Score distribution
            scores = [review.get('score', 0) for review in self.reviews]
            plt.figure(figsize=(10, 6))
            plt.hist(scores, bins=range(1, 7), alpha=0.7, edgecolor='black')
            plt.xlabel('Rating')
            plt.ylabel('Frequency')
            plt.title(f'Rating Distribution - {app_name}')
            plt.xticks(range(1, 6))
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{self.output_dir}/rating_distribution_{app_name.lower().replace(' ', '_')}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Thumbs up distribution
            thumbs_up = [review.get('thumbsUpCount', 0) for review in self.reviews]
            if any(thumbs_up):
                plt.figure(figsize=(10, 6))
                plt.hist(thumbs_up, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Thumbs Up Count')
                plt.ylabel('Frequency')
                plt.title(f'Thumbs Up Distribution - {app_name}')
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{self.output_dir}/thumbs_up_distribution_{app_name.lower().replace(' ', '_')}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info("Visualizations created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
    
    def run(self, app_name: str = "app"):
        """Run complete analysis pipeline"""
        processed_reviews = self.preprocess_reviews()
        
        if processed_reviews:
            # Generate word frequency analysis first
            self.analyze_word_frequency(processed_reviews, app_name)
            
            # Generate wordclouds
            self.generate_wordcloud(processed_reviews, app_name)
            
            # Topic modeling
            self.topic_modeling(processed_reviews, app_name)
        
        # Sentiment analysis (works with original reviews)
        self.sentiment_analysis(app_name)
        
        # Problem analysis - NEW FEATURE
        problem_stats = self.analyze_user_problems(app_name)
        if problem_stats:
            self.generate_problem_recommendations(problem_stats, app_name)
        
        # Create visualizations
        self.create_visualizations(app_name)
        
        self.logger.info(f"Review processing completed for {app_name}")
    
    def analyze_word_frequency(self, processed_reviews: List[str], app_name: str = "app"):
        """Comprehensive word frequency analysis"""
        if not processed_reviews:
            self.logger.warning("No processed reviews for word frequency analysis")
            return
            
        try:
            from collections import Counter
            import numpy as np
            
            # Combine all text
            all_text = ' '.join(processed_reviews)
            words = all_text.split()
            
            # Word frequency analysis
            word_freq = Counter(words)
            total_words = len(words)
            unique_words = len(word_freq)
            
            # Statistical analysis
            frequencies = list(word_freq.values())
            
            analysis_data = {
                'app_name': app_name,
                'total_words': total_words,
                'unique_words': unique_words,
                'vocabulary_richness': unique_words / total_words if total_words > 0 else 0,
                'most_common_word': word_freq.most_common(1)[0] if word_freq else ('', 0),
                'average_word_frequency': np.mean(frequencies) if frequencies else 0,
                'median_word_frequency': np.median(frequencies) if frequencies else 0,
                'top_50_words': word_freq.most_common(50)
            }
            
            # Save detailed analysis
            output_path = f"{self.output_dir}/word_frequency_analysis_{app_name.lower().replace(' ', '_')}.json"
            
            # Convert Counter to serializable format
            serializable_data = analysis_data.copy()
            serializable_data['top_50_words'] = [[word, freq] for word, freq in analysis_data['top_50_words']]
            serializable_data['most_common_word'] = list(analysis_data['most_common_word'])
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Word frequency analysis saved to {output_path}")
            
            # Create comprehensive visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Top 15 words bar chart
            top_15 = word_freq.most_common(15)
            words_15, freq_15 = zip(*top_15) if top_15 else ([], [])
            
            bars1 = ax1.bar(range(len(words_15)), freq_15, color='lightcoral', edgecolor='darkred')
            ax1.set_xlabel('Words', fontweight='bold')
            ax1.set_ylabel('Frequency', fontweight='bold')
            ax1.set_title(f'Top 15 Most Frequent Words - {app_name}', fontweight='bold')
            ax1.set_xticks(range(len(words_15)))
            ax1.set_xticklabels(words_15, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, freq in zip(bars1, freq_15):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{freq}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Word length distribution
            word_lengths = [len(word) for word in words]
            ax2.hist(word_lengths, bins=range(1, max(word_lengths) + 2), alpha=0.7, color='skyblue', edgecolor='navy')
            ax2.set_xlabel('Word Length', fontweight='bold')
            ax2.set_ylabel('Frequency', fontweight='bold')
            ax2.set_title(f'Word Length Distribution - {app_name}', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Frequency distribution (Zipf's law visualization)
            ranks = range(1, min(101, len(word_freq) + 1))
            top_100_frequencies = [freq for _, freq in word_freq.most_common(100)]
            
            ax3.loglog(ranks[:len(top_100_frequencies)], top_100_frequencies, 'bo-', alpha=0.7)
            ax3.set_xlabel('Word Rank (log scale)', fontweight='bold')
            ax3.set_ylabel('Frequency (log scale)', fontweight='bold')
            ax3.set_title(f'Word Frequency vs Rank (Zipf\'s Law) - {app_name}', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Cumulative frequency
            sorted_frequencies = sorted(frequencies, reverse=True)
            cumulative_freq = np.cumsum(sorted_frequencies)
            cumulative_percent = (cumulative_freq / total_words) * 100
            
            ax4.plot(range(1, len(cumulative_percent) + 1), cumulative_percent, 'g-', linewidth=2)
            ax4.set_xlabel('Number of Unique Words', fontweight='bold')
            ax4.set_ylabel('Cumulative Frequency (%)', fontweight='bold')
            ax4.set_title(f'Cumulative Word Frequency - {app_name}', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% coverage')
            ax4.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% coverage')
            ax4.legend()
            
            plt.tight_layout()
            viz_output_path = f"{self.output_dir}/word_frequency_comprehensive_{app_name.lower().replace(' ', '_')}.png"
            plt.savefig(viz_output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Comprehensive word frequency visualization saved to {viz_output_path}")
            
            # Print analysis summary
            print(f"\n📈 WORD FREQUENCY ANALYSIS - {app_name}")
            print("=" * 50)
            print(f"Total Words: {total_words:,}")
            print(f"Unique Words: {unique_words:,}")
            print(f"Vocabulary Richness: {analysis_data['vocabulary_richness']:.4f}")
            print(f"Most Common Word: '{analysis_data['most_common_word'][0]}' ({analysis_data['most_common_word'][1]} times)")
            print(f"Average Word Frequency: {analysis_data['average_word_frequency']:.2f}")
            print(f"Median Word Frequency: {analysis_data['median_word_frequency']:.2f}")
            print("=" * 50)
            
            return analysis_data
            
        except Exception as e:
            self.logger.error(f"Error in word frequency analysis: {e}")
            return None

    def analyze_user_problems(self, app_name: str = "app"):
        """Comprehensive user problem analysis and categorization"""
        if not self.reviews:
            self.logger.warning("No reviews for problem analysis")
            return
            
        try:
            # Define problem categories with Indonesian keywords
            problem_categories = {
                'Login_Authentication': {
                    'keywords': ['login', 'masuk', 'sign in', 'password', 'akun', 'account', 'sandi', 'lupa password', 'forgot password', 'tidak bisa masuk', 'gabisa login', 'ga bisa masuk'],
                    'description': 'Masalah login, autentikasi, dan akses akun'
                },
                'Performance_Speed': {
                    'keywords': ['lambat', 'slow', 'lemot', 'loading', 'lag', 'ngelag', 'patah patah', 'not responding', 'hang', 'freeze', 'stuck', 'macet'],
                    'description': 'Masalah performa dan kecepatan aplikasi'
                },
                'Crashes_Errors': {
                    'keywords': ['crash', 'error', 'force close', 'berhenti sendiri', 'keluar sendiri', 'tutup sendiri', 'tidak merespon', 'gagal', 'failed', 'bug', 'rusak'],
                    'description': 'Aplikasi crash, error, atau berhenti bekerja'
                },
                'Network_Connection': {
                    'keywords': ['internet', 'koneksi', 'connection', 'network', 'offline', 'tidak terhubung', 'no connection', 'jaringan', 'sinyal'],
                    'description': 'Masalah koneksi internet dan jaringan'
                },
                'User_Interface': {
                    'keywords': ['tampilan', 'ui', 'interface', 'design', 'layout', 'button', 'tombol', 'menu', 'tidak terlihat', 'terpotong', 'overlap'],
                    'description': 'Masalah tampilan dan antarmuka pengguna'
                },
                'Features_Functionality': {
                    'keywords': ['fitur', 'feature', 'fungsi', 'tidak berfungsi', 'tidak bekerja', 'hilang', 'missing', 'gak ada', 'ga ada', 'tidak ada'],
                    'description': 'Masalah fitur dan fungsi aplikasi'
                },
                'Data_Sync': {
                    'keywords': ['sinkronisasi', 'sync', 'data', 'tidak tersimpan', 'hilang data', 'lost data', 'backup', 'restore'],
                    'description': 'Masalah sinkronisasi dan kehilangan data'
                },
                'Payment_Transaction': {
                    'keywords': ['pembayaran', 'payment', 'transaksi', 'bayar', 'saldo', 'transfer', 'top up', 'withdraw', 'failed payment'],
                    'description': 'Masalah pembayaran dan transaksi'
                },
                'Notification_Alerts': {
                    'keywords': ['notifikasi', 'notification', 'pemberitahuan', 'alert', 'tidak muncul notif', 'ga ada notif'],
                    'description': 'Masalah notifikasi dan peringatan'
                },
                'Customer_Service': {
                    'keywords': ['customer service', 'cs', 'support', 'bantuan', 'help', 'contact', 'komplain', 'laporan', 'respon lambat'],
                    'description': 'Masalah layanan pelanggan dan dukungan'
                },
                'Security_Privacy': {
                    'keywords': ['keamanan', 'security', 'privacy', 'privasi', 'data pribadi', 'hack', 'spam', 'phishing'],
                    'description': 'Masalah keamanan dan privasi'
                },
                'Installation_Update': {
                    'keywords': ['install', 'update', 'upgrade', 'download', 'gagal install', 'failed update', 'tidak bisa update'],
                    'description': 'Masalah instalasi dan pembaruan aplikasi'
                }
            }
            
            # Analyze each review for problems
            problem_data = []
            category_counts = {category: 0 for category in problem_categories.keys()}
            category_counts['Other_Issues'] = 0
            category_counts['No_Problem_Detected'] = 0
            
            for i, review in enumerate(self.reviews):
                content = review.get('content', '').lower()
                score = review.get('score', 5)
                
                detected_problems = []
                problem_keywords_found = []
                
                # Only analyze low-rated reviews (1-3 stars) for problems
                if score <= 3 and content:
                    for category, category_data in problem_categories.items():
                        keywords_found = []
                        for keyword in category_data['keywords']:
                            if keyword in content:
                                keywords_found.append(keyword)
                        
                        if keywords_found:
                            detected_problems.append(category)
                            problem_keywords_found.extend(keywords_found)
                            category_counts[category] += 1
                    
                    # If no specific problem detected but low rating, categorize as "Other_Issues"
                    if not detected_problems:
                        detected_problems.append('Other_Issues')
                        category_counts['Other_Issues'] += 1
                else:
                    # High-rated reviews (4-5 stars) are considered no problem
                    detected_problems.append('No_Problem_Detected')
                    category_counts['No_Problem_Detected'] += 1
                
                problem_data.append({
                    'review_index': i,
                    'score': score,
                    'content_preview': content[:100] + '...' if len(content) > 100 else content,
                    'detected_problems': detected_problems,
                    'keywords_found': problem_keywords_found,
                    'problem_count': len([p for p in detected_problems if p not in ['No_Problem_Detected']])
                })
            
            # Create problem analysis DataFrame
            df_problems = pd.DataFrame(problem_data)
            
            # Save detailed problem analysis
            output_path = f"{self.output_dir}/problem_analysis_{app_name.lower().replace(' ', '_')}.csv"
            df_problems.to_csv(output_path, index=False)
            
            # Calculate problem statistics
            total_reviews = len(self.reviews)
            low_rated_reviews = sum(1 for r in self.reviews if r.get('score', 5) <= 3)
            
            problem_stats = {
                'app_name': app_name,
                'analysis_date': datetime.now().isoformat(),
                'total_reviews_analyzed': total_reviews,
                'low_rated_reviews': low_rated_reviews,
                'problem_detection_rate': (low_rated_reviews / total_reviews * 100) if total_reviews > 0 else 0,
                'category_distribution': {},
                'top_problems': [],
                'problem_severity': 'Low'
            }
            
            # Calculate percentages and sort by frequency
            for category, count in category_counts.items():
                if category not in ['No_Problem_Detected']:
                    percentage = (count / total_reviews * 100) if total_reviews > 0 else 0
                    problem_stats['category_distribution'][category] = {
                        'count': count,
                        'percentage': round(percentage, 2),
                        'description': problem_categories.get(category, {}).get('description', 'Masalah lainnya')
                    }
            
            # Sort problems by frequency
            sorted_problems = sorted(
                [(cat, data['count'], data['percentage']) for cat, data in problem_stats['category_distribution'].items()],
                key=lambda x: x[1], reverse=True
            )
            
            problem_stats['top_problems'] = [
                {
                    'category': cat,
                    'count': count,
                    'percentage': pct,
                    'description': problem_categories.get(cat, {}).get('description', 'Masalah lainnya')
                }
                for cat, count, pct in sorted_problems[:10]
            ]
            
            # Determine problem severity
            if problem_stats['problem_detection_rate'] > 30:
                problem_stats['problem_severity'] = 'High'
            elif problem_stats['problem_detection_rate'] > 15:
                problem_stats['problem_severity'] = 'Medium'
            else:
                problem_stats['problem_severity'] = 'Low'
            
            # Save problem statistics
            stats_output_path = f"{self.output_dir}/problem_statistics_{app_name.lower().replace(' ', '_')}.json"
            with open(stats_output_path, 'w', encoding='utf-8') as f:
                json.dump(problem_stats, f, ensure_ascii=False, indent=2)
            
            # Create problem visualization
            self.create_problem_visualizations(category_counts, problem_categories, app_name)
            
            # Print problem analysis summary
            print(f"\n🚨 USER PROBLEM ANALYSIS - {app_name}")
            print("=" * 60)
            print(f"Total Reviews Analyzed: {total_reviews:,}")
            print(f"Low-Rated Reviews (1-3 ⭐): {low_rated_reviews:,}")
            print(f"Problem Detection Rate: {problem_stats['problem_detection_rate']:.1f}%")
            print(f"Problem Severity Level: {problem_stats['problem_severity']}")
            print("\n📊 TOP PROBLEMS IDENTIFIED:")
            print("-" * 60)
            
            for i, problem in enumerate(problem_stats['top_problems'][:10], 1):
                if problem['count'] > 0:
                    print(f"{i:2d}. {problem['category'].replace('_', ' '):<25} : {problem['count']:>3} reviews ({problem['percentage']:>5.1f}%)")
                    print(f"    └─ {problem['description']}")
                    print()
            
            print("=" * 60)
            
            self.logger.info(f"Problem analysis completed and saved to {output_path}")
            return problem_stats
            
        except Exception as e:
            self.logger.error(f"Error in problem analysis: {e}")
            return None
    
    def create_problem_visualizations(self, category_counts: Dict, problem_categories: Dict, app_name: str):
        """Create visualizations for problem analysis"""
        try:
            # Filter out zero counts and non-problem categories
            filtered_counts = {k: v for k, v in category_counts.items() 
                             if v > 0 and k not in ['No_Problem_Detected']}
            
            if not filtered_counts:
                self.logger.warning("No problems detected for visualization")
                return
            
            # Create comprehensive problem visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Problem frequency bar chart
            categories = list(filtered_counts.keys())
            counts = list(filtered_counts.values())
            
            # Sort by count
            sorted_data = sorted(zip(categories, counts), key=lambda x: x[1], reverse=True)
            categories, counts = zip(*sorted_data) if sorted_data else ([], [])
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            bars1 = ax1.bar(range(len(categories)), counts, color=colors, edgecolor='black', alpha=0.8)
            ax1.set_xlabel('Problem Categories', fontweight='bold')
            ax1.set_ylabel('Number of Reports', fontweight='bold')
            ax1.set_title(f'User Problems Frequency - {app_name}', fontweight='bold', fontsize=14)
            ax1.set_xticks(range(len(categories)))
            ax1.set_xticklabels([cat.replace('_', '\n') for cat in categories], rotation=45, ha='right', fontsize=10)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, count in zip(bars1, counts):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
            
            # 2. Problem distribution pie chart
            if len(categories) > 0:
                ax2.pie(counts, labels=[cat.replace('_', ' ') for cat in categories], autopct='%1.1f%%', 
                       colors=colors, startangle=90)
                ax2.set_title(f'Problem Distribution - {app_name}', fontweight='bold', fontsize=14)
            
            # 3. Problem severity heatmap
            severity_matrix = []
            severity_labels = []
            
            for category in categories:
                if category in problem_categories:
                    # Create severity score based on frequency
                    severity_score = filtered_counts[category]
                    severity_matrix.append([severity_score])
                    severity_labels.append(category.replace('_', ' '))
            
            if severity_matrix:
                im = ax3.imshow(severity_matrix, cmap='Reds', aspect='auto')
                ax3.set_yticks(range(len(severity_labels)))
                ax3.set_yticklabels(severity_labels, fontsize=10)
                ax3.set_xticks([0])
                ax3.set_xticklabels(['Severity'], fontweight='bold')
                ax3.set_title(f'Problem Severity Matrix - {app_name}', fontweight='bold', fontsize=14)
                
                # Add text annotations
                for i, severity in enumerate(severity_matrix):
                    ax3.text(0, i, f'{severity[0]}', ha='center', va='center', fontweight='bold', color='white')
            
            # 4. Problem trend (if we have review dates)
            review_dates = [r.get('at') for r in self.reviews if r.get('at')]
            if review_dates and len(review_dates) > 10:
                # Group by month and count low-rated reviews
                df_dates = pd.DataFrame({'date': review_dates, 'score': [r.get('score', 5) for r in self.reviews]})
                df_dates['month'] = pd.to_datetime(df_dates['date']).dt.to_period('M')
                monthly_problems = df_dates[df_dates['score'] <= 3].groupby('month').size()
                
                if len(monthly_problems) > 1:
                    monthly_problems.plot(kind='line', ax=ax4, marker='o', linewidth=2, markersize=6)
                    ax4.set_xlabel('Month', fontweight='bold')
                    ax4.set_ylabel('Problem Reports', fontweight='bold')
                    ax4.set_title(f'Problem Reports Trend - {app_name}', fontweight='bold', fontsize=14)
                    ax4.grid(True, alpha=0.3)
                    ax4.tick_params(axis='x', rotation=45)
                else:
                    ax4.text(0.5, 0.5, 'Insufficient data\nfor trend analysis', 
                            ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                    ax4.set_title(f'Problem Trend - {app_name}', fontweight='bold', fontsize=14)
            else:
                ax4.text(0.5, 0.5, 'No date data available\nfor trend analysis', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title(f'Problem Trend - {app_name}', fontweight='bold', fontsize=14)
            
            plt.tight_layout()
            viz_output_path = f"{self.output_dir}/problem_analysis_visualization_{app_name.lower().replace(' ', '_')}.png"
            plt.savefig(viz_output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create a separate detailed problem breakdown chart
            if len(categories) > 0:
                plt.figure(figsize=(14, 8))
                
                # Horizontal bar chart for better readability
                y_pos = np.arange(len(categories))
                bars = plt.barh(y_pos, counts, color=colors, edgecolor='black', alpha=0.8)
                
                plt.xlabel('Number of Problem Reports', fontweight='bold', fontsize=12)
                plt.ylabel('Problem Categories', fontweight='bold', fontsize=12)
                plt.title(f'Detailed User Problem Breakdown - {app_name}', fontweight='bold', fontsize=16)
                plt.yticks(y_pos, [cat.replace('_', ' ') for cat in categories])
                
                # Add value labels and descriptions
                for i, (bar, category, count) in enumerate(zip(bars, categories, counts)):
                    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                            f'{count}', va='center', fontweight='bold')
                    
                    # Add description as subtitle
                    if category in problem_categories:
                        description = problem_categories[category]['description']
                        plt.text(0.02 * max(counts), bar.get_y() + bar.get_height()/2,
                               f'{description}', va='center', fontsize=9, alpha=0.7)
                
                plt.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                
                detailed_viz_path = f"{self.output_dir}/problem_breakdown_detailed_{app_name.lower().replace(' ', '_')}.png"
                plt.savefig(detailed_viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Problem visualizations saved: {viz_output_path}, {detailed_viz_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating problem visualizations: {e}")
    
    def generate_problem_recommendations(self, problem_stats: Dict, app_name: str):
        """Generate actionable recommendations based on problem analysis"""
        try:
            recommendations = {
                'app_name': app_name,
                'analysis_date': datetime.now().isoformat(),
                'overall_assessment': '',
                'priority_actions': [],
                'detailed_recommendations': {},
                'improvement_roadmap': []
            }
            
            # Generate overall assessment
            severity = problem_stats.get('problem_severity', 'Low')
            detection_rate = problem_stats.get('problem_detection_rate', 0)
            
            if severity == 'High':
                recommendations['overall_assessment'] = f"URGENT: Aplikasi mengalami masalah serius dengan tingkat deteksi masalah {detection_rate:.1f}%. Diperlukan tindakan segera."
            elif severity == 'Medium':
                recommendations['overall_assessment'] = f"PERHATIAN: Aplikasi memiliki beberapa masalah dengan tingkat deteksi {detection_rate:.1f}%. Perlu perbaikan prioritas."
            else:
                recommendations['overall_assessment'] = f"BAIK: Aplikasi dalam kondisi relatif stabil dengan tingkat masalah {detection_rate:.1f}%."
            
            # Generate recommendations based on top problems
            recommendation_templates = {
                'Login_Authentication': {
                    'priority': 'HIGH',
                    'actions': [
                        'Perbaiki sistem autentikasi dan reset password',
                        'Implementasi two-factor authentication',
                        'Upgrade server authentication untuk mengurangi downtime'
                    ]
                },
                'Performance_Speed': {
                    'priority': 'HIGH',
                    'actions': [
                        'Optimasi kode dan database queries',
                        'Implementasi caching dan CDN',
                        'Upgrade infrastruktur server'
                    ]
                },
                'Crashes_Errors': {
                    'priority': 'CRITICAL',
                    'actions': [
                        'Debug dan fix critical bugs',
                        'Implementasi crash reporting yang lebih baik',
                        'Extensive testing sebelum release'
                    ]
                },
                'Network_Connection': {
                    'priority': 'MEDIUM',
                    'actions': [
                        'Implementasi offline mode',
                        'Optimasi penggunaan bandwidth',
                        'Better error handling untuk koneksi buruk'
                    ]
                },
                'User_Interface': {
                    'priority': 'MEDIUM',
                    'actions': [
                        'UI/UX redesign untuk elemen bermasalah',
                        'Responsive design improvements',
                        'User testing untuk validasi changes'
                    ]
                },
                'Features_Functionality': {
                    'priority': 'HIGH',
                    'actions': [
                        'Audit lengkap semua fitur',
                        'Fix broken features prioritas tinggi',
                        'Documentation dan tutorial yang lebih baik'
                    ]
                }
            }
            
            # Process top problems
            top_problems = problem_stats.get('top_problems', [])
            for problem in top_problems[:5]:  # Top 5 problems
                category = problem['category']
                count = problem['count']
                percentage = problem['percentage']
                
                if category in recommendation_templates:
                    template = recommendation_templates[category]
                    
                    recommendations['detailed_recommendations'][category] = {
                        'problem_description': problem['description'],
                        'affected_users': f"{count} reviews ({percentage}%)",
                        'priority': template['priority'],
                        'recommended_actions': template['actions'],
                        'estimated_impact': 'High' if percentage > 10 else 'Medium' if percentage > 5 else 'Low'
                    }
                    
                    # Add to priority actions if critical
                    if template['priority'] in ['CRITICAL', 'HIGH'] and percentage > 5:
                        recommendations['priority_actions'].append({
                            'category': category,
                            'action': template['actions'][0],  # First action as priority
                            'urgency': template['priority'],
                            'affected_percentage': percentage
                        })
            
            # Generate improvement roadmap
            if recommendations['priority_actions']:
                recommendations['improvement_roadmap'] = [
                    {
                        'phase': 'Immediate (0-2 weeks)',
                        'actions': [action['action'] for action in recommendations['priority_actions'] if action['urgency'] == 'CRITICAL']
                    },
                    {
                        'phase': 'Short-term (2-8 weeks)',
                        'actions': [action['action'] for action in recommendations['priority_actions'] if action['urgency'] == 'HIGH']
                    },
                    {
                        'phase': 'Medium-term (2-6 months)',
                        'actions': ['UI/UX improvements', 'Feature enhancements', 'Performance optimizations']
                    },
                    {
                        'phase': 'Long-term (6+ months)',
                        'actions': ['Major infrastructure upgrades', 'New feature development', 'Platform expansion']
                    }
                ]
            
            # Save recommendations
            rec_output_path = f"{self.output_dir}/problem_recommendations_{app_name.lower().replace(' ', '_')}.json"
            with open(rec_output_path, 'w', encoding='utf-8') as f:
                json.dump(recommendations, f, ensure_ascii=False, indent=2)
            
            # Print recommendations summary
            print(f"\n💡 IMPROVEMENT RECOMMENDATIONS - {app_name}")
            print("=" * 60)
            print(f"Overall Assessment: {recommendations['overall_assessment']}")
            print("\n🚨 PRIORITY ACTIONS:")
            for i, action in enumerate(recommendations['priority_actions'][:5], 1):
                print(f"{i}. [{action['urgency']}] {action['action']}")
                print(f"   └─ Affects {action['affected_percentage']:.1f}% of problem reports")
            
            print("\n🗓️ IMPROVEMENT ROADMAP:")
            for phase in recommendations['improvement_roadmap']:
                if phase['actions']:
                    print(f"\n{phase['phase']}:")
                    for action in phase['actions'][:3]:  # Limit to 3 actions per phase
                        print(f"  • {action}")
            
            print("=" * 60)
            
            self.logger.info(f"Problem recommendations saved to {rec_output_path}")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return None

    # ...existing code...
def analyze_app_reviews(app_id: str, app_name: str, max_reviews_per_star: int = 100):
    """Comprehensive analysis for a single app"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting comprehensive analysis for {app_name}")
    
    all_reviews = []
    
    # Fetch reviews for each star rating
    for star in range(1, 6):
        scraper = GooglePlayReviewScraper(
            app_id=app_id,
            app_name=app_name,
            score=star,
            max_reviews=max_reviews_per_star
        )
        reviews = scraper.run()
        all_reviews.extend(reviews)
    
    # Process all reviews together
    if all_reviews:
        processor = ReviewProcessor(all_reviews)
        processor.run(app_name)
        
        # Generate comprehensive report
        generate_comprehensive_report(all_reviews, app_name)
    else:
        logger.warning(f"No reviews found for {app_name}")

def generate_comprehensive_report(reviews: List[Dict], app_name: str):
    """Generate a comprehensive analysis report"""
    logger = logging.getLogger(__name__)
    
    try:
        # Calculate comprehensive statistics
        total_reviews = len(reviews)
        avg_score = sum(r.get('score', 0) for r in reviews) / total_reviews if total_reviews > 0 else 0
        
        score_dist = {}
        for i in range(1, 6):
            score_dist[f"{i}_star"] = sum(1 for r in reviews if r.get('score') == i)
            score_dist[f"{i}_star_percentage"] = (score_dist[f"{i}_star"] / total_reviews) * 100 if total_reviews > 0 else 0
        
        # Content analysis
        contents = [r.get('content', '') for r in reviews if r.get('content')]
        avg_content_length = sum(len(content.split()) for content in contents) / len(contents) if contents else 0
        
        # Date analysis
        dates = [r.get('at') for r in reviews if r.get('at')]
        if dates:
            latest_date = max(dates)
            earliest_date = min(dates)
            date_range = (latest_date - earliest_date).days
        else:
            latest_date = earliest_date = date_range = None
        
        report = {
            'app_name': app_name,
            'analysis_date': datetime.now().isoformat(),
            'total_reviews': total_reviews,
            'average_score': round(avg_score, 2),
            'score_distribution': score_dist,
            'content_stats': {
                'reviews_with_content': len(contents),
                'average_content_length_words': round(avg_content_length, 2)
            },
            'date_range': {
                'earliest_review': earliest_date.isoformat() if earliest_date else None,
                'latest_review': latest_date.isoformat() if latest_date else None,
                'span_days': date_range
            },
            'engagement_stats': {
                'total_thumbs_up': sum(r.get('thumbsUpCount', 0) for r in reviews),
                'avg_thumbs_up': sum(r.get('thumbsUpCount', 0) for r in reviews) / total_reviews if total_reviews > 0 else 0,
                'reviews_with_replies': sum(1 for r in reviews if r.get('replyContent'))
            }
        }
        
        # Save comprehensive report
        output_path = f"reviews/comprehensive_report_{app_name.lower().replace(' ', '_')}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Comprehensive report saved to {output_path}")
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"COMPREHENSIVE ANALYSIS SUMMARY - {app_name}")
        print(f"{'='*50}")
        print(f"Total Reviews: {total_reviews}")
        print(f"Average Score: {avg_score:.2f}/5.0")
        print(f"Score Distribution:")
        for i in range(5, 0, -1):
            count = score_dist[f"{i}_star"]
            percentage = score_dist[f"{i}_star_percentage"]
            print(f"  {i} ⭐: {count:4d} ({percentage:5.1f}%)")
        print(f"Average Content Length: {avg_content_length:.1f} words")
        print(f"Total Thumbs Up: {report['engagement_stats']['total_thumbs_up']}")
        print(f"Reviews with Replies: {report['engagement_stats']['reviews_with_replies']}")
        if date_range:
            print(f"Review Date Range: {date_range} days")
        print(f"{'='*50}\n")
        
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {e}")

if __name__ == "__main__":
    # Configuration
    apps = [
        {"id": "id.go.kemnaker.siapkerja", "name": "SIAP Kerja"},
    ]
    
    # You can adjust this number based on your needs
    MAX_REVIEWS_PER_STAR = 50  # Reduced for faster processing
    
    print("🚀 Starting Enhanced Google Play Review Analysis")
    print(f"📱 Analyzing {len(apps)} apps with up to {MAX_REVIEWS_PER_STAR} reviews per star rating")
    print("-" * 60)
    
    for app in apps:
        try:
            analyze_app_reviews(app["id"], app["name"], MAX_REVIEWS_PER_STAR)
        except Exception as e:
            logging.error(f"Failed to analyze {app['name']}: {e}")
            continue
    
    print("✅ Analysis completed! Check the 'reviews' folder for all outputs.")
    print("📊 Generated files include:")
    print("   - Raw data: TXT, CSV, JSON files")
    print("   - Visualizations: Word clouds, rating distributions")
    print("   - Analysis: Sentiment analysis, topic modeling")
    print("   - Reports: Comprehensive analysis reports")