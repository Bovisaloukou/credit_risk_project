import pandas as pd
import numpy as np
import os

def generate_credit_data(num_records=100000, output_dir='data', filename='generated_credit_data.csv'):
    """
    Génère un dataset synthétique pour la prédiction du risque de crédit.
    """
    output_path = os.path.join(output_dir, filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Génération de {num_records} enregistrements...")

    # Dictionnaire pour stocker les variables
    features = {}

    # Caractéristiques de base
    np.random.seed(42) # Pour la reproductibilité
    features['income'] = np.random.normal(loc=50000, scale=30000, size=num_records)
    features['income'][features['income'] < 10000] = 10000 # Salaire minimum
    features['income'] = np.round(features['income'], 2) # Arrondir

    features['loan_amount'] = np.random.normal(loc=15000, scale=10000, size=num_records)
    features['loan_amount'][features['loan_amount'] < 1000] = 1000 # Montant minimum du prêt
    features['loan_amount'] = np.round(features['loan_amount'], 2) # Arrondir

    # Ajout d'une corrélation simple : prêts plus élevés pour revenus plus élevés
    features['loan_amount'] = features['loan_amount'] + (features['income'] / 5.0) # Simple correlation
    features['loan_amount'] = np.round(features['loan_amount'], 2) # Arrondir à nouveau

    features['payment_delays'] = np.random.randint(0, 10, size=num_records) # Nombre de retards

    features['employment_length'] = np.random.normal(loc=5, scale=3, size=num_records) # en années
    features['employment_length'][features['employment_length'] < 0] = 0 # Pas d'ancienneté négative
    features['employment_length'][features['employment_length'] > 30] = 30 # Maximum
    features['employment_length'] = np.round(features['employment_length'], 1) # Arrondir

    features['age'] = np.random.randint(18, 70, size=num_records)

    # Introduire des valeurs manquantes
    for col in ['income', 'employment_length']:
        missing_indices = np.random.choice(num_records, size=int(num_records * 0.05), replace=False) # 5% missing
        features[col][missing_indices] = np.nan

    # Plus de valeurs manquantes pour payment_delays
    missing_delays_indices = np.random.choice(num_records, size=int(num_records * 0.15), replace=False) # 15% missing
    features['payment_delays'] = features['payment_delays'].astype(float) # Permet d'assigner NaN
    features['payment_delays'][missing_delays_indices] = np.nan

    # Génération de la cible (risque de défaut)
    loan_to_income_ratio_base = np.round(features['loan_amount'] / features['income'], 2)
    loan_to_income_ratio_base[np.isnan(loan_to_income_ratio_base)] = 0.5

    age_effect = np.abs(features['age'] - 45)

    # Coefficients
    beta_loan_to_income = 1.5
    beta_payment_delays = 0.5
    beta_employment_length = -0.4
    beta_age_effect = 0.05
    intercept = -3.5

    # Calcul du Logit
    temp_payment_delays = np.nan_to_num(features['payment_delays'], nan=np.nanmedian(features['payment_delays']))
    temp_employment_length = np.nan_to_num(features['employment_length'], nan=np.nanmedian(features['employment_length']))

    logit = (intercept +
             beta_loan_to_income * loan_to_income_ratio_base +
             beta_payment_delays * temp_payment_delays +
             beta_employment_length * temp_employment_length +
             beta_age_effect * age_effect)

    # Convertir le logit en probabilité
    probability_of_default = 1 / (1 + np.exp(-logit))
    probability_of_default = np.clip(probability_of_default, 0.01, 0.99)

    # Générer la cible binaire
    features['default'] = (np.random.rand(num_records) < probability_of_default).astype(int)

    # Créer le DataFrame
    data = pd.DataFrame(features)

    # Sauvegarder le DataFrame
    data.to_csv(output_path, index=False)

    print(f"Données générées et sauvegardées dans {output_path}")
    print(f"Taux de défaut généré: {data['default'].mean():.2%}")

if __name__ == "__main__":
    generate_credit_data()