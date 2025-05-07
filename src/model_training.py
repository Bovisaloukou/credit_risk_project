import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Importer les composants locaux
from src.preprocessing import build_preprocessing_pipeline, NUMERICAL_COLS, ArrayToDataFrameTransformer
from src.features import FeatureEngineeringTransformer
from src.data_generation import generate_credit_data # Pour s'assurer que les données existent


# Chemins des fichiers
DATA_FILEPATH = 'data/generated_credit_data.csv'
MODEL_DIR = 'models'
PIPELINE_FILEPATH = os.path.join(MODEL_DIR, 'full_pipeline.joblib')
# Note: Avec un pipeline complet, scaler et imputer sont inclus. Pas besoin de les sauvegarder séparément.

# Paramètres de Grid Search (simplifiés pour l'exemple)
# En pratique, une grille plus large et plus de CV seraient nécessaires.
PARAM_DIST = {
    'estimator__n_estimators': [100, 200, 300, 400],
    'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'estimator__max_depth': [3, 4, 5, 6],
    'estimator__subsample': [0.7, 0.8, 1.0],
    'estimator__min_samples_split': [2, 5, 10, 20]
}


def load_data(filepath):
    """Charge les données."""
    print(f"Chargement des données depuis {filepath}...")
    try:
        data = pd.read_csv(filepath)
        print("Données chargées avec succès.")
        print(f"Dimensions: {data.shape}")
        # Vérifier si les colonnes essentielles sont présentes
        expected_cols = NUMERICAL_COLS + ['default']
        if not all(col in data.columns for col in expected_cols):
            missing = [col for col in expected_cols if col not in data.columns]
            print(f"Erreur: Colonnes attendues manquantes: {missing}")
            return None
        return data
    except FileNotFoundError:
        print(f"Erreur: Fichier non trouvé à {filepath}")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return None


def build_full_pipeline(imputer_type='median'):
    """
    Construit le pipeline complet de prétraitement, d'ingénierie des caractéristiques et de modélisation.
    Permet de choisir le type d'imputation ('median' ou 'knn').
    """
    # 1. Pipeline de prétraitement (Imputation et Scaling) - Sortie: numpy array
    preprocessor = build_preprocessing_pipeline(imputer_type=imputer_type)

    # 2. Convertir l'array numpy en DataFrame - Sortie: DataFrame avec noms NUMERICAL_COLS
    array_to_df_converter = ArrayToDataFrameTransformer(column_names=NUMERICAL_COLS)

    # 3. Ingénierie des caractéristiques - Sortie: DataFrame avec nouvelles features
    feature_engineer = FeatureEngineeringTransformer()

    # 4. Modèle final
    estimator = GradientBoostingClassifier(random_state=42)

    # Construire le pipeline complet
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),              # Numpy Array (cols numériques seulement)
        ('array_to_df', array_to_df_converter),      # DataFrame (avec noms NUMERICAL_COLS)
        ('feature_engineer', feature_engineer),      # DataFrame (avec nouvelles features)
        ('estimator', estimator)                     # Modèle (prend le DataFrame)
    ])

    return pipeline

def train_model(pipeline, X_train, y_train, param_dist=None, n_iter=20):
    """
    Entraîne le pipeline complet avec RandomizedSearchCV.
    """
    print("Début de l'entraînement du pipeline...")

    if param_dist is None:
         param_dist = PARAM_DIST # Utiliser la distribution définie en haut du fichier
         print("Utilisation de la distribution de paramètres par défaut pour RandomizedSearchCV.")
    else:
         print("Utilisation de la distribution de paramètres fournie pour RandomizedSearchCV.")


    # Initialiser RandomizedSearchCV
    # Note: Les noms des paramètres dans param_dist doivent correspondre aux noms des étapes du pipeline + nom_param
    # ex: 'estimator__n_estimators' car 'estimator' est le nom de l'étape GradientBoostingClassifier
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=5,  # Validation croisée en 5 plis
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    # Entraîner RandomizedSearchCV
    print("Lancement de RandomizedSearchCV...")
    # Fit le pipeline complet, y compris preprocessing et FE, sur les données d'entraînement
    search.fit(X_train, y_train)

    # Récupérer le meilleur estimateur (qui est le pipeline complet ajusté)
    best_pipeline = search.best_estimator_
    print("RandomizedSearchCV terminé.")
    print(f"Meilleurs paramètres trouvés: {search.best_params_}")
    print(f"Meilleur score ROC AUC (validation): {search.best_score_:.4f}")

    print("Entraînement du pipeline complet terminé.")
    return best_pipeline

def evaluate_model(pipeline, X_test, y_test):
    """
    Évalue le pipeline entraîné sur les données de test.
    """
    if pipeline is None:
        print("Erreur: Pipeline non entraîné ou chargé. Impossible d'évaluer.")
        return None

    print("Début de l'évaluation du pipeline...")
    # predict_proba et predict appliquent automatiquement tout le pipeline (preprocess, FE, model)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1] # Probabilité de la classe positive (défaut=1)
    y_pred = pipeline.predict(X_test) # Prédictions binaires (seuil 0.5 par défaut)

    # Calculer les métriques
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Précision (Precision): {precision:.4f}")
    print(f"Rappel (Recall): {recall:.4f}")
    print("Matrice de Confusion:")
    print(conf_matrix)

    evaluation_metrics = {
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix.tolist() # Convertir en liste
    }

    print("Évaluation terminée.")
    return evaluation_metrics

def save_pipeline(pipeline, filepath=PIPELINE_FILEPATH):
    """
    Sauvegarde le pipeline complet entraîné.
    """
    if pipeline is None:
        print("Erreur: Pipeline non entraîné. Rien à sauvegarder.")
        return

    # S'assurer que le répertoire models existe
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print(f"Sauvegarde du pipeline dans {filepath}...")
    try:
        joblib.dump(pipeline, filepath)
        print("Pipeline sauvegardé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du pipeline: {e}")


# --- Bloc d'exécution principal pour l'entraînement ---
if __name__ == "__main__":
    # 1. Générer les données (si elles n'existent pas)
    if not os.path.exists(DATA_FILEPATH):
        generate_credit_data(output_dir='data', filename='generated_credit_data.csv')
    else:
        print(f"Le fichier de données {DATA_FILEPATH} existe déjà. Chargement...")

    # 2. Charger les données
    data = load_data(DATA_FILEPATH)

    if data is not None:
        # 3. Séparer les caractéristiques (X) et la cible (y)
        X = data.drop('default', axis=1)
        y = data['default']

        # 4. Séparer les données en ensembles d'entraînement et de test
        # Utiliser uniquement les colonnes d'origine pour le split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        print(f"Taille de l'ensemble d'entraînement: {X_train.shape}")
        print(f"Taille de l'ensemble de test: {X_test.shape}")

        # 5. Construire le pipeline (choix de l'imputation possible)
        full_pipeline = build_full_pipeline(imputer_type='knn')
        print("\nPipeline construit:")
        print(full_pipeline)

        # 6. Entraîner le pipeline (avec RandomizedSearchCV inclus)
        trained_pipeline = train_model(full_pipeline, X_train, y_train)

        # 7. Évaluer le pipeline entraîné sur l'ensemble de test
        evaluation_results = evaluate_model(trained_pipeline, X_test, y_test)
        print("\nRésultats de l'évaluation sur l'ensemble de test:")
        print(evaluation_results)

        # 8. Sauvegarder le pipeline entraîné
        save_pipeline(trained_pipeline)
    else:
        print("Impossible de charger les données. Entraînement annulé.")