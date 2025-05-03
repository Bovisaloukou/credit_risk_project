import joblib
import pandas as pd
import numpy as np
import os

# Importer les classes custom nécessaires pour que joblib puisse les charger
from src.preprocessing import ArrayToDataFrameTransformer, NUMERICAL_COLS # Importer le transformer de conversion DF
from src.features import DebtToIncomeTransformer, AgeRiskTransformer, PaymentHistoryScoreTransformer, FeatureEngineeringTransformer # Importer les transformers de FE

# Chemin du pipeline sauvegardé
PIPELINE_FILEPATH = 'models/full_pipeline.joblib'

class CreditRiskPredictionService:
    """
    Service pour charger le pipeline ML et faire des prédictions.
    """
    def __init__(self, pipeline_filepath=PIPELINE_FILEPATH):
        self.pipeline_filepath = pipeline_filepath
        self.pipeline = None
        self._load_pipeline() # Charger le pipeline à l'initialisation

    def _load_pipeline(self):
        """
        Charge le pipeline ML sauvegardé.
        """
        if not os.path.exists(self.pipeline_filepath):
            print(f"Erreur: Fichier du pipeline introuvable à {self.pipeline_filepath}.")
            print("Veuillez d'abord exécuter src/model_training.py pour entraîner et sauvegarder le pipeline.")
            self.pipeline = None # S'assurer que self.pipeline est None en cas d'échec
            return

        print(f"Chargement du pipeline depuis {self.pipeline_filepath}...")
        try:
            # joblib.load a besoin d'accéder aux définitions des classes custom (ArrayToDataFrameTransformer, FeatureEngineeringTransformer, etc.)
            # C'est pourquoi nous les importons en haut de ce fichier.
            self.pipeline = joblib.load(self.pipeline_filepath)
            print("Pipeline chargé avec succès.")
        except Exception as e:
            print(f"Erreur lors du chargement du pipeline: {e}")
            self.pipeline = None


    def predict_proba(self, raw_data):
        """
        Prédit la probabilité de défaut pour de nouvelles données brutes.

        Args:
            raw_data (pd.DataFrame ou dict ou list of dict): Les nouvelles données brutes
                                                             pour la prédiction.
                                                             Doit contenir les colonnes brutes attendues.

        Returns:
            np.ndarray: Probabilités de défaut pour chaque enregistrement,
                        ou None si le pipeline n'est pas chargé ou si les données sont invalides.
        """
        if self.pipeline is None:
            print("Erreur: Pipeline non chargé. Impossible de prédire.")
            return None

        # S'assurer que raw_data est un DataFrame
        if isinstance(raw_data, dict):
            # Convertir un dictionnaire (souvent une seule ligne) en DataFrame
            # Gérer le cas où les valeurs ne sont pas déjà des listes
            data_dict_for_df = {k: [v] if not isinstance(v, list) else v for k, v in raw_data.items()}
            data_df = pd.DataFrame(data_dict_for_df)
        elif isinstance(raw_data, list):
             # Convertir une liste de dictionnaires en DataFrame
             try:
                  data_df = pd.DataFrame(raw_data)
             except Exception as e:
                  print(f"Erreur: Impossible de convertir la liste de dictionnaires en DataFrame: {e}")
                  return None
        elif isinstance(raw_data, pd.DataFrame):
            data_df = raw_data.copy() # Utiliser une copie pour éviter de modifier l'original
        else:
            print("Erreur: Les données de prédiction doivent être un DataFrame, un dictionnaire ou une liste de dictionnaires.")
            return None

        # Supprimer la colonne 'default' si présente (ne doit pas être passée au pipeline)
        if 'default' in data_df.columns:
             data_df = data_df.drop('default', axis=1)
             print("Note: Colonne 'default' supprimée des données de prédiction.")

        # Vérifier que les colonnes d'entrée nécessaires sont présentes avant de passer au pipeline
        # Le pipeline s'attend aux colonnes spécifiées dans build_preprocessing_pipeline (NUMERICAL_COLS + remainder si pas drop)
        # Dans notre cas, ColumnTransformer avec remainder='drop' s'attend aux NUMERICAL_COLS en entrée.
        expected_input_cols = NUMERICAL_COLS # Colonnes brutes attendues en entrée du pipeline
        if not all(col in data_df.columns for col in expected_input_cols):
            missing_cols = [col for col in expected_input_cols if col not in data_df.columns]
            print(f"Erreur: Données d'entrée manquantes. Colonnes attendues: {expected_input_cols}. Manquantes: {missing_cols}.")
            return None

        # Sélectionner les colonnes d'entrée dans le bon ordre (important pour le ColumnTransformer)
        data_for_pipeline = data_df[expected_input_cols]

        print("Début de la prédiction avec le pipeline...")
        try:
            # Utiliser le pipeline chargé pour faire la prédiction
            # predict_proba appliquera toutes les étapes du pipeline séquentiellement
            probabilities = self.pipeline.predict_proba(data_for_pipeline)[:, 1] # Probabilité de la classe positive (défaut=1)
            print("Prédiction terminée.")
            return probabilities
        except Exception as e:
            print(f"Erreur lors de l'exécution du pipeline de prédiction: {e}")
            # Log plus de détails sur l'erreur si possible
            import traceback
            traceback.print_exc()
            return None

if __name__ == '__main__':
    # Exemple d'utilisation du service de prédiction
    # Assurez-vous que le pipeline a été entraîné et sauvegardé (en exécutant src/model_training.py)

    service = CreditRiskPredictionService()

    if service.pipeline is not None:
        # Créer des données d'exemple pour la prédiction
        new_client_data_single = {
            "income": 75000.0,
            "loan_amount": 30000.0,
            "payment_delays": 1,
            "employment_length": 10.0,
            "age": 45
        }

        new_clients_data_multiple = [
            new_client_data_single,
            {
                "income": 35000.0,
                "loan_amount": 22000.0,
                "payment_delays": 4,
                "employment_length": 2.5,
                "age": 28
            },
             {
                "income": 90000.0,
                "loan_amount": 15000.0,
                "payment_delays": 0,
                "employment_length": 18.0,
                "age": 55
            }
        ]

        print("\nPrédiction pour un seul client (dictionnaire):")
        probabilities_single = service.predict_proba(new_client_data_single)
        if probabilities_single is not None:
             print(f"Probabilité de défaut: {probabilities_single[0]:.4f}")

        print("\nPrédiction pour plusieurs clients (liste de dictionnaires):")
        probabilities_multiple = service.predict_proba(new_clients_data_multiple)
        if probabilities_multiple is not None:
             print(f"Probabilités de défaut: {probabilities_multiple.tolist()}")

        print("\nPrédiction avec un DataFrame:")
        data_df_test = pd.DataFrame(new_clients_data_multiple)
        probabilities_df = service.predict_proba(data_df_test)
        if probabilities_df is not None:
            print(f"Probabilités de défaut: {probabilities_df.tolist()}")

    else:
        print("\nService de prédiction non initialisé car pipeline non chargé.")