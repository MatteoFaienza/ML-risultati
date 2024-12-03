import pandas as pd


file_path = "results.csv"
df_results = pd.read_csv(file_path)

home_team_counts = df_results['home_team'].value_counts()
away_team_counts = df_results['away_team'].value_counts()

total_counts = home_team_counts.add(away_team_counts, fill_value=0)
sorted_total_counts = total_counts.sort_values(ascending=False)

print(sorted_total_counts.head(60))