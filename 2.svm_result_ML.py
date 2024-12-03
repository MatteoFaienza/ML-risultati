
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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

#print(df_italy_results.head(60))


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


# Trasformo la data da object a datetime
df_italy_results['date'] = pd.to_datetime(df_italy_results['date'])


# Divido la data in mesi e anno
df_italy_results['year'] = df_italy_results['date'].dt.year
df_italy_results['month'] = df_italy_results['date'].dt.month

df_italy_results = df_italy_results.drop("date", axis = 1)

#print(df_italy_results.head(60))

min_max = MinMaxScaler()
label_enc = LabelEncoder()

for column in df_italy_results.columns:
    
    if df_italy_results[column].dtype in ['int64', 'float64']:
        df_italy_results[column] = min_max.fit_transform(df_italy_results[[column]]) 
        
    elif df_italy_results[column].dtype == 'object' or df_italy_results[column].dtype == 'bool':
        df_italy_results[column] = label_enc.fit_transform(df_italy_results[column])

#print(df_italy_results.head(60))

X = df_italy_results[["home_team","away_team","home_score","away_score","tournament","city","country","neutral","year","month"]]
y = df_italy_results["Match result"]

# 5.1 Split dei dati
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2)

random_for = RandomForestClassifier(n_estimators=100, random_state=42)

# Alleno il medellino
random_for.fit(X_train, y_train)

# Prediction
y_pred = random_for.predict(X_test)

# Accuracy model
accuracy = accuracy_score(y_test, y_pred)
report_class = classification_report(y_test, y_pred)
print("Accuracy ", accuracy )
print(report_class)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

