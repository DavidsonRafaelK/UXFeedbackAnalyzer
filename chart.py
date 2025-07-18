import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Simulasi data dari hasil akhir sebelumnya
data = {
    'ux_category': [
        'Benefit Issue', 'Connectivity Issue', 'Login Issue', 'Navigation Issue',
        'Other UX Issue', 'Performance Issue', 'Technical Issue', 'Top Up Issue'
    ],
    'Negative': [299, 9078, 1586, 569, 4777, 13438, 6942, 412],
    'Neutral': [86, 1117, 281, 134, 1375, 1630, 1024, 77],
    'Positive': [818, 3018, 269, 294, 4667, 1406, 588, 333]
}
df = pd.DataFrame(data)

# Pie chart (total per category)
df['total'] = df[['Negative', 'Neutral', 'Positive']].sum(axis=1)
plt.figure(figsize=(8, 8))
plt.pie(df['total'], labels=df['ux_category'], autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title("Distribution of UX Categories")
plt.tight_layout()
plt.savefig("images/piechart.png")
plt.close()

# Heatmap (percentage sentiment per category)
df_percent = df[['Negative', 'Neutral', 'Positive']].div(df['total'], axis=0) * 100
plt.figure(figsize=(10, 6))
sns.heatmap(df_percent, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Percentage'})
plt.title("Sentiment Distribution per UX Category (%)")
plt.xlabel("Sentiment")
plt.ylabel("UX Category")
plt.tight_layout()
plt.savefig("images/heatmap.png")
plt.close()
