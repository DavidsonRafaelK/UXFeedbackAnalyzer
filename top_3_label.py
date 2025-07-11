import pandas as pd

def top_3_label_reviews(df):
    # Get the top 3 labeled reviews by count
    return df['ux_category'].value_counts()
    

def get_all_other_ux_issues(df):
    return  df[df['ux_category'] == 'Other UX Issue']

def group_by_ux_category(df):
    return df.groupby(['ux_category', 'sentiment']).size().unstack().fillna(0)

if __name__ == "__main__":
    list_of_csv = [
        'reviews/filter/ux_label_1_star_mytelkomsel.csv',
        'reviews/filter/ux_label_2_star_mytelkomsel.csv',
        'reviews/filter/ux_label_3_star_mytelkomsel.csv',
        'reviews/filter/ux_label_4_star_mytelkomsel.csv',
        'reviews/filter/ux_label_5_star_mytelkomsel.csv'
    ]

    all_data = pd.DataFrame()

    for csv_file in list_of_csv:
        df = pd.read_csv(csv_file)
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    top_labels = top_3_label_reviews(all_data)
    other_issues = get_all_other_ux_issues(all_data)
    grouped_data = group_by_ux_category(all_data)
    
    print("Top 3 UX Categories:")
    print(top_labels)
    print("Other UX Issues:")
    print(other_issues)
    print("Grouped UX Issues:")
    print(grouped_data)