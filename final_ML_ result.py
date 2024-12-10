
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import BaggingClassifier

file_path = "results.csv"
df_results = pd.read_csv(file_path)

#print(df_results.info())

#print(df_results.isna().sum().sum())

#print(df_results.describe())

df_results["home_team"].unique()

team_italy = "Italy"

def italy_word(columns, word):
    
    return word.lower() in columns["home_team"].lower() or word.lower() in columns["away_team"].lower()

df_italy_results = df_results[df_results.apply(italy_word, axis=1 ,word = team_italy)]


def result_of_the_match (columns):

    if columns['home_team'] == 'Italy':
        if columns['home_score'] > columns['away_score']:
            return 'Victory'
        elif columns['home_score'] < columns['away_score']:
            return "Lose"
        else:
            return "Tie"
        
    elif columns['away_team'] == 'Italy':
        if columns['away_score'] > columns['home_score']:
            return 'Victory'
        elif columns['away_score'] < columns['home_score']:
            return "Lose"
        else:
            return "Tie"

# Aggiungo la colonna risultato 
df_italy_results["Match result"] = df_italy_results.apply(result_of_the_match, axis=1)

# Ordina per data
df_italy_results['date'] = pd.to_datetime(df_italy_results['date'])
df_italy_results = df_italy_results.sort_values(by='date').reset_index(drop=True)

# Aggiungo le 5 partite preceedenti
last_res = df_italy_results['Match result'].tolist()
df_italy_results['previous_result'] = last_res
df_italy_results['previous_result'] = df_italy_results['previous_result'].shift(periods=1)


last_home_score = df_italy_results['home_score'].tolist()
last_away_score = df_italy_results['away_score'].tolist()

df_italy_results['last_home_score'] = last_home_score
df_italy_results['last_away_score'] = last_away_score

df_italy_results['last_home_score'] = df_italy_results['last_home_score'].shift(periods=1)
df_italy_results['last_away_score'] = df_italy_results['last_away_score'].shift(periods=1)



last_res = df_italy_results['Match result'].tolist()
df_italy_results['2_previous_result'] = last_res
df_italy_results['2_previous_result'] = df_italy_results['2_previous_result'].shift(periods=2)


last_home_score = df_italy_results['home_score'].tolist()
last_away_score = df_italy_results['away_score'].tolist()

df_italy_results['2_last_home_score'] = last_home_score
df_italy_results['2_last_away_score'] = last_away_score

df_italy_results['2_last_home_score'] = df_italy_results['2_last_home_score'].shift(periods=2)
df_italy_results['2_last_away_score'] = df_italy_results['2_last_away_score'].shift(periods=2)



last_res = df_italy_results['Match result'].tolist()
df_italy_results['3_previous_result'] = last_res
df_italy_results['3_previous_result'] = df_italy_results['3_previous_result'].shift(periods=3)


last_home_score = df_italy_results['home_score'].tolist()
last_away_score = df_italy_results['away_score'].tolist()

df_italy_results['3_last_home_score'] = last_home_score
df_italy_results['3_last_away_score'] = last_away_score

df_italy_results['3_last_home_score'] = df_italy_results['3_last_home_score'].shift(periods=3)
df_italy_results['3_last_away_score'] = df_italy_results['3_last_away_score'].shift(periods=3)



last_res = df_italy_results['Match result'].tolist()
df_italy_results['4_previous_result'] = last_res
df_italy_results['4_previous_result'] = df_italy_results['4_previous_result'].shift(periods=4)


last_home_score = df_italy_results['home_score'].tolist()
last_away_score = df_italy_results['away_score'].tolist()

df_italy_results['4_last_home_score'] = last_home_score
df_italy_results['4_last_away_score'] = last_away_score

df_italy_results['4_last_home_score'] = df_italy_results['4_last_home_score'].shift(periods=4)
df_italy_results['4_last_away_score'] = df_italy_results['4_last_away_score'].shift(periods=4)


last_res = df_italy_results['Match result'].tolist()
df_italy_results['5_previous_result'] = last_res
df_italy_results['5_previous_result'] = df_italy_results['5_previous_result'].shift(periods=5)


last_home_score = df_italy_results['home_score'].tolist()
last_away_score = df_italy_results['away_score'].tolist()

df_italy_results['5_last_home_score'] = last_home_score
df_italy_results['5_last_away_score'] = last_away_score

df_italy_results['5_last_home_score'] = df_italy_results['5_last_home_score'].shift(periods=5)
df_italy_results['5_last_away_score'] = df_italy_results['5_last_away_score'].shift(periods=5)


df_italy_results = df_italy_results.tail(df_italy_results.shape[0]-5)


df_italy_results['year'] = df_italy_results['date'].dt.year
df_italy_results['month'] = df_italy_results['date'].dt.month
df_italy_results = df_italy_results.sort_values(by='year').reset_index(drop=True)

df_italy_results = df_italy_results.drop("date", axis = 1)
df_italy_results = df_italy_results.dropna().reset_index(drop=True)

min_max = MinMaxScaler()
label_enc = LabelEncoder()

for column in df_italy_results.columns:
    
    if df_italy_results[column].dtype in ['int64', 'float64']:
        df_italy_results[column] = min_max.fit_transform(df_italy_results[[column]]) 
        
    elif df_italy_results[column].dtype == 'object' or df_italy_results[column].dtype == 'bool':
        df_italy_results[column] = label_enc.fit_transform(df_italy_results[column])


# Feature e target
X = df_italy_results[["home_team", "away_team",
                      "tournament", "city", "country", "neutral",
                      "previous_result", "last_home_score", "last_away_score", 
                      "2_previous_result", "2_last_home_score", "2_last_away_score",
                      "3_previous_result", "3_last_home_score", "3_last_away_score",
                      "4_previous_result", "4_last_home_score", "4_last_away_score",
                      "5_previous_result", "5_last_home_score", "5_last_away_score"]]

y = df_italy_results["Match result"]

# Split dei dati
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2)

rmf = RandomForestClassifier(n_estimators=100, random_state=42)
model = BaggingClassifier(estimator=rmf,n_estimators=100, random_state=0)


# Alleno il medellino
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy model
accuracy = accuracy_score(y_test, y_pred)
report_class = classification_report(y_test, y_pred)
print("Accuracy ", accuracy )
print(report_class)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

