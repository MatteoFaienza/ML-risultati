import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carica i dati
file_path = "results.csv"
df_results = pd.read_csv(file_path)

# Funzione per selezionare le partite dell'Italia
team_italy = "Italy"

def italy_word(columns, word):
    return word.lower() in columns["home_team"].lower() or word.lower() in columns["away_team"].lower()

# Filtro i risultati delle partite dell'Italia
df_italy_results = df_results[df_results.apply(italy_word, axis=1, word=team_italy)].copy()

# Funzione per determinare il risultato della partita
def result_of_the_match(columns):
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

# Aggiungo la colonna "Match result"
df_italy_results.loc[:, "Match result"] = df_italy_results.apply(result_of_the_match, axis=1)

# Visualizza i primi 5 valori della colonna 'date' per diagnosticare il formato
print("Prime righe della colonna 'date':")
print(df_italy_results['date'].head())

# Controllo i tipi di dato della colonna 'date'
print("Tipo di dato prima della conversione:", df_italy_results['date'].dtype)

# Provo a forzare la conversione in datetime
df_italy_results.loc[:, 'date'] = pd.to_datetime(df_italy_results['date'], errors='coerce')

# Verifica se ci sono valori NaT dopo la conversione e segnala eventuali problemi
if df_italy_results['date'].isnull().any():
    print("Attenzione: ci sono valori invalidi in 'date' che sono stati convertiti in NaT.")

# Verifica di nuovo il tipo di dato
print(f"Tipo di dato della colonna 'date' dopo la conversione: {df_italy_results['date'].dtype}")

# Se la conversione è avvenuta correttamente, estraggo anno e mese
if df_italy_results['date'].dtype == 'datetime64[ns]':
    df_italy_results.loc[:, 'year'] = df_italy_results['date'].dt.year
    df_italy_results.loc[:, 'month'] = df_italy_results['date'].dt.month
else:
    print("Errore: la colonna 'date' non è stata convertita correttamente in datetime.")

# A questo punto puoi continuare con la parte successiva del codice per rimuovere 'date' e procedere con l'analisi
df_italy_results = df_italy_results.drop("date", axis=1)

# Ordino per anno decrescente e prendo le ultime 5 partite
df_italy_results = df_italy_results.sort_values(by='year', ascending=False)
df_italy_results = df_italy_results.head(5)

# Mostra la struttura finale
print(df_italy_results.head())
