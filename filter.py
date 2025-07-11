import pandas as pd

class Labeling_UX_Reviews:
    def __init__(self, df):
        self.df = df

    def cleaning_reviews(self):
        """Clean the reviews DataFrame by removing nulls, stripping whitespace, filter date just 2025 and duplicates."""
        df = self.df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'].notnull() & (df['date'].dt.year == 2025)]
        df = df[df['message'].notnull() & df['message'].str.strip() != '']
        df['message'] = df['message'].str.lower()
        df = df.drop_duplicates(subset=['message'])
        self.df = df
        return df

    def label_reviews(self):
        self.df['ux_category'] = self.df.apply(self.label_ux_reviews, axis=1)
        return self.df
    
    def ux_keywords(self):
        return {
            'Technical Issue': [
                'error', 'bug', 'hang', 'crash', 'tidak ada', 'tidak jalan', 'masalah teknis', 'gangguan', 'tidak berfungsi', 'tidak bisa', 'tidak muncul', 'tidak tampil', 'tidak dapat dibuka'
            ],
            'Performance Issue': [
                'lemot', 'lambat', 'lama', 'muter', 'berat', 'lelet', 'animasi', 'kinerja', 'buffering', 'memuat', 'kinerja', 'fitur', 'tidak responsif'
            ],
            'Connectivity Issue': [
                'sinyal', 'koneksi', 'akses', 'jaringan', 'internet', 'data', 'gangguan', 'kuota', 'tidak stabil'
            ],
            'Login Issue': [
                'login', 'logout', 'masuk', 'keluar', 'akun', 'password', 'gagal masuk', 'tidak bisa masuk', 'verifikasi'
            ],
            'Navigation Issue': [
                'navigasi', 'pusing', 'susah', 'bingung', 'tampilan', 'menu', 'interface', 'tidak memudahkan', 'akses', 'user interface', 'antarmuka'
            ],
            'Top Up Issue': [
                'top up', 'saldo', 'bayar', 'pembayaran', 'isi ulang', 'transaksi', 'gagal top up', 'payment', 'kredit', 'debit'
            ],
            'Benefit Issue': [
                'promo', 'benefit', 'penawaran', 'reward', 'tidak jelas', 'tidak sesuai', 'poin', 'voucher', 'diskon', 'hadiah'
            ]
        }

    def label_ux_reviews(self, row):
        msg = str(row['message']).lower()
        score = row['score']

        if score >= 4:
            if any(kyw in msg for kyw in ['bagus', 'baik', 'suka', 'puas', 'terbaik', 'mantap', 'mudah', 'praktis', 'cepat', 'efisien',
                                          'memuaskan', 'user friendly', 'simpel', 'keren', 'membantu', 'rekomendasi', 'enak', 'berguna', 'cocok']):
                return "Positive Feedback"

        for category, keywords in self.ux_keywords().items():
            for keyword in keywords:
                if keyword in msg:
                    return category
        return "Other UX Issue"
    
    def save_labeled_reviews(self, filename):
        self.df.to_csv(filename, index=False)
        return f"Labeled reviews saved to {filename} with {len(self.df)} entries."
    
    def classify_sentiment(self, row):
        negative_keywords = ['jelek', 'buruk', 'masalah', 'kecewa', 'gagal', 'bermasalah', 'lambat', 'lemot', 'bug', 'loading']
        score = row['score']
        message = str(row['message']).lower()

        if score <= 2:
            return "Negative"
        elif score == 3:
            return "Neutral"
        elif score >= 4:
            for keyword in negative_keywords:
                if keyword in message:
                    return "Negative"
            return "Positive"

if __name__ == "__main__":
    list_of_csv = [
        'reviews/data_mentah/csv/mytelkomsel/reviews_1_star_mytelkomsel.csv',
        'reviews/data_mentah/csv/mytelkomsel/reviews_2_star_mytelkomsel.csv',
        'reviews/data_mentah/csv/mytelkomsel/reviews_3_star_mytelkomsel.csv',
        'reviews/data_mentah/csv/mytelkomsel/reviews_4_star_mytelkomsel.csv',
        'reviews/data_mentah/csv/mytelkomsel/reviews_5_star_mytelkomsel.csv'
    ]

    output_of_csv = [
        'reviews/filter/ux_label_1_star_mytelkomsel.csv',
        'reviews/filter/ux_label_2_star_mytelkomsel.csv',
        'reviews/filter/ux_label_3_star_mytelkomsel.csv',
        'reviews/filter/ux_label_4_star_mytelkomsel.csv',
        'reviews/filter/ux_label_5_star_mytelkomsel.csv'
    ]

    for input_csv, output_csv in zip(list_of_csv, output_of_csv):
        df = pd.read_csv(input_csv)
        labeling = Labeling_UX_Reviews(df)
        cleaned_df = labeling.cleaning_reviews()
        labeled_df = labeling.label_reviews()
        labeled_df['sentiment'] = labeled_df.apply(labeling.classify_sentiment, axis=1)
        result = labeling.save_labeled_reviews(output_csv)

        print(result)
        print(f"Processed {input_csv} -> Total label reviews: {len(labeled_df)} and saved to {output_csv}.")

