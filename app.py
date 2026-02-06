import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- CHARGEMENT DES FICHIERS ---
@st.cache_resource
def load_objects():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model_columns.pkl', 'rb') as f:
        cols = pickle.load(f)
    return model, scaler, cols

model, scaler, model_columns = load_objects()

# --- INTERFACE STREAMLIT ---
st.title("🩺 Analyse et Prédiction des Coûts Hospitaliers")
st.markdown("""
Cette application permet d'estimer le montant de la facturation médicale en fonction du profil du patient
et de sa condition médicale.
""")

# --- SECTION 1 : DESCRIPTION DES DONNÉES (Ce que le prof exige) ---
st.header("📊 Description des données")
st.write("""
Le modèle a été entraîné sur un jeu de données de santé comprenant des informations sur :
- **Le profil patient** : Âge, Sexe, Groupe Sanguin.
- **Le séjour** : Durée d'hospitalisation, Type d'admission (Urgent, Elective, Emergency).
- **Le traitement** : Maladie diagnostiquée, Médication prescrite et résultats des tests.
""")

# --- SECTION 2 : FORMULAIRE DE PRÉDICTION ---
st.header("🔮 Simulation de facturation")
st.info("Remplissez les informations ci-dessous pour obtenir une estimation.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Âge du patient", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Genre", ["Male", "Female"])
    duration = st.number_input("Durée du séjour (jours)", min_value=1, value=5)
    condition = st.selectbox("Condition Médicale", ["Cancer", "Obesity", "Diabetes", "Asthma", "Hypertension", "Arthritis"])

with col2:
    blood = st.selectbox("Groupe Sanguin", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    admission = st.selectbox("Type d'Admission", ["Urgent", "Emergency", "Elective"])
    medication = st.selectbox("Médication", ["Paracetamol", "Ibuprofen", "Aspirin", "Penicillin", "Lipitor"])
    test_res = st.selectbox("Résultat du Test", ["Normal", "Abnormal", "Inconclusive"])

# --- BOUTON DE PRÉDICTION ---
if st.button("Calculer l'estimation"):
    # 1. Créer un DataFrame vide avec les colonnes du modèle
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
   
    # 2. Remplir les données numériques
    input_df['Age'] = age
    input_df['Duration_Days'] = duration
   
    # 3. Gérer les colonnes encodées (One-Hot Encoding manuel)
    # On met 1 dans la colonne correspondante si elle existe dans le modèle
    for col in [f"Gender_{gender}", f"Blood Type_{blood}", f"Medical Condition_{condition}",
                f"Admission Type_{admission}", f"Medication_{medication}", f"Test Results_{test_res}"]:
        if col in model_columns:
            input_df[col] = 1
           
    # 4. Scaling
    input_scaled = scaler.transform(input_df)
   
    # 5. Prédiction
    prediction = model.predict(input_scaled)
   
    st.success(f"### Estimation du montant : {prediction[0]:,.2f} $")

