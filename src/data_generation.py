import pandas as pd
import numpy as np
import os

def generate_credit_data(num_records=1000000, output_dir='data', filename='generated_credit_data.csv'):
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

    features['employment_length'] = np.random.normal(loc=5, scale=3, size=num_records)
    features['employment_length'][features['employment_length'] < 0] = 0 # Pas d'ancienneté négative
    features['employment_length'][features['employment_length'] > 30] = 30 # Maximum
    features['employment_length'] = np.round(features['employment_length'], 1) # Arrondir

    features['age'] = np.random.randint(18, 70, size=num_records)

    # Ajouter des caractéristiques supplémentaires
    features['credit_score'] = np.random.randint(300, 850, size=num_records) # Score de crédit
    features['debt_to_income_ratio'] = np.round(features['loan_amount'] / features['income'], 2) # Ratio dette/revenu
    features['debt_to_income_ratio'][features['debt_to_income_ratio'] > 1] = 1 # Limiter à 1

    # Type d'emploi (catégoriel)
    employment_types = ['CDI', 'CDD', 'Indépendant', 'Sans emploi', 'Retraité', 'Fonctionnaire']
    features['employment_type'] = np.random.choice(employment_types, size=num_records, p=[0.45, 0.15, 0.15, 0.10, 0.10, 0.05])

    # Historique de paiement (score de 0 à 100)
    features['payment_history_score'] = np.random.normal(loc=70, scale=20, size=num_records)
    features['payment_history_score'] = np.clip(features['payment_history_score'], 0, 100)

    # Situation matrimoniale
    marital_statuses = ['Célibataire', 'Marié', 'Divorcé', 'Veuf']
    features['marital_status'] = np.random.choice(marital_statuses, size=num_records, p=[0.35, 0.50, 0.10, 0.05])

    # Nombre de personnes à charge
    features['num_dependents'] = np.random.poisson(lam=1.2, size=num_records)
    features['num_dependents'] = np.clip(features['num_dependents'], 0, 5)

    # Propriété du logement
    home_ownership = ['Propriétaire', 'Locataire', 'Logé par famille']
    features['home_ownership'] = np.random.choice(home_ownership, size=num_records, p=[0.55, 0.40, 0.05])

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
    beta_credit_score = -0.01 # Plus le score est élevé, moins le risque est élevé
    intercept = -1.5  # Ajusté pour viser un taux de défaut d'environ 5 %

    # Calcul du Logit
    temp_payment_delays = np.nan_to_num(features['payment_delays'], nan=np.nanmedian(features['payment_delays']))
    temp_employment_length = np.nan_to_num(features['employment_length'], nan=np.nanmedian(features['employment_length']))

    logit = (intercept +
             beta_loan_to_income * loan_to_income_ratio_base +
             beta_payment_delays * temp_payment_delays +
             beta_employment_length * temp_employment_length +
             beta_age_effect * age_effect +
             beta_credit_score * features['credit_score'])

    # Mettre à jour le calcul du logit pour inclure les nouvelles variables
    beta_payment_history = -0.02  # Plus le score est élevé, moins le risque est élevé
    beta_employment_type = {'CDI': -0.3, 'CDD': 0.1, 'Indépendant': 0.2, 'Sans emploi': 0.5, 'Retraité': 0.1, 'Fonctionnaire': -0.2}
    beta_num_dependents = 0.08
    beta_home_ownership = {'Propriétaire': -0.1, 'Locataire': 0.1, 'Logé par famille': 0.2}

    temp_payment_history = features['payment_history_score']
    temp_employment_type = np.array([beta_employment_type[et] for et in features['employment_type']])
    temp_num_dependents = features['num_dependents']
    temp_home_ownership = np.array([beta_home_ownership[ho] for ho in features['home_ownership']])

    logit = (logit +
             beta_payment_history * temp_payment_history +
             temp_employment_type +
             beta_num_dependents * temp_num_dependents +
             temp_home_ownership)

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