import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Charger le modèle
MODEL_PATH = 'models/full_pipeline.joblib'
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Le modèle n'a pas été trouvé. Veuillez entraîner le modèle et placer le fichier dans le dossier 'models'.")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("Évaluation du Risque de Crédit")
st.write("Remplissez les informations du client pour obtenir une prédiction du risque de défaut.")

# Interface utilisateur pour saisir les caractéristiques
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Revenu annuel (€)", min_value=10000, max_value=500000, value=50000, step=1000)
    loan_amount = st.number_input("Montant du prêt (€)", min_value=1000, max_value=200000, value=15000, step=500)
    payment_delays = st.number_input("Nombre de retards de paiement", min_value=0, max_value=20, value=0, step=1)
    employment_length = st.number_input("Ancienneté (années)", min_value=0.0, max_value=40.0, value=5.0, step=0.5)
    age = st.number_input("Âge", min_value=18, max_value=80, value=35, step=1)
    credit_score = st.number_input("Score de crédit", min_value=300, max_value=850, value=650, step=1)
    debt_to_income_ratio = st.slider("Ratio dette/revenu", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

with col2:
    payment_history_score = st.slider("Score d'historique de paiement", min_value=0, max_value=100, value=70, step=1)
    employment_type = st.selectbox("Type d'emploi", ["CDI", "CDD", "Indépendant", "Sans emploi", "Retraité", "Fonctionnaire"])
    marital_status = st.selectbox("Situation matrimoniale", ["Célibataire", "Marié", "Divorcé", "Veuf"])
    num_dependents = st.number_input("Nombre de personnes à charge", min_value=0, max_value=10, value=0, step=1)
    home_ownership = st.selectbox("Statut de logement", ["Propriétaire", "Locataire", "Logé par famille"])

# Prédiction
if st.button("Prédire le risque de défaut"):
    if model is None:
        st.stop()
    input_dict = {
        'income': income,
        'loan_amount': loan_amount,
        'payment_delays': payment_delays,
        'employment_length': employment_length,
        'age': age,
        'credit_score': credit_score,
        'debt_to_income_ratio': debt_to_income_ratio,
        'payment_history_score': payment_history_score,
        'employment_type': employment_type,
        'marital_status': marital_status,
        'num_dependents': num_dependents,
        'home_ownership': home_ownership
    }
    input_df = pd.DataFrame([input_dict])
    try:
        proba = model.predict_proba(input_df)[0, 1]
        st.success(f"Probabilité de défaut : {proba*100:.2f} %")
        if proba > 0.5:
            st.error("Risque ÉLEVÉ de défaut !")
        elif proba > 0.2:
            st.warning("Risque MODÉRÉ de défaut.")
        else:
            st.info("Risque FAIBLE de défaut.")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
