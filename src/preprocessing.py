import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin # Importer aussi pour ArrayToDataFrameTransformer


# Colonnes numériques et catégorielles enrichies
NUMERICAL_COLS = [
    'income', 'loan_amount', 'payment_delays', 'employment_length', 'age',
    'credit_score', 'debt_to_income_ratio', 'payment_history_score', 'num_dependents'
]
CATEGORICAL_COLS = [
    'employment_type', 'marital_status', 'home_ownership'
]

def build_preprocessing_pipeline(imputer_type='median'):
    """
    Construit un pipeline de prétraitement pour les caractéristiques numériques et catégorielles.
    imputer_type: 'median' ou 'knn'
    """
    if imputer_type == 'knn':
        numerical_imputer = KNNImputer(n_neighbors=5)
    else:
        numerical_imputer = SimpleImputer(strategy='median')

    numerical_pipeline = Pipeline([
        ('imputer', numerical_imputer),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, NUMERICAL_COLS),
            ('cat', categorical_pipeline, CATEGORICAL_COLS)
        ],
        remainder='drop'
    )
    return preprocessor

class ArrayToDataFrameTransformer(BaseEstimator, TransformerMixin):
    """
    Convertit un numpy array en DataFrame, en restaurant les noms de colonnes.
    Assume que l'array provient du ColumnTransformer sur NUMERICAL_COLS
    avec remainder='drop'.
    """
    def __init__(self, column_names=NUMERICAL_COLS): # Utilise NUMERICAL_COLS par défaut
        self.column_names = column_names

    def fit(self, X, y=None):
        # Peut éventuellement vérifier que X (qui est un numpy array) a la bonne forme
        if isinstance(X, np.ndarray):
             if X.shape[1] != len(self.column_names):
                  print(f"Attention: ArrayToDataFrameTransformer - Nombre de colonnes de l'array ({X.shape[1]}) ne correspond pas aux noms attendus ({len(self.column_names)}).")
             self.n_features_in_ = X.shape[1] # Stocke le nombre de features en entrée
        else:
             # Devrait recevoir un numpy array du ColumnTransformer
             raise TypeError("ArrayToDataFrameTransformer attend un numpy array.")
        return self

    def transform(self, X):
        """
        Convertit l'array X en DataFrame avec les noms de colonnes spécifiés.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("ArrayToDataFrameTransformer attend un numpy array pour la transformation.")

        # Créer un DataFrame à partir de l'array en utilisant les noms de colonnes
        # Gérer le cas où le nombre de colonnes ne correspond pas exactement (si ColumnTransformer a changé)
        # Pour un pipeline correct, X.shape[1] DEVRAIT être len(self.column_names) si remainder='drop'.
        # Sinon, il y a un problème dans la construction du pipeline.
        if X.shape[1] != len(self.column_names):
             raise ValueError(f"ArrayToDataFrameTransformer: Le nombre de colonnes de l'array ({X.shape[1]}) ne correspond pas au nombre de noms de colonnes ({len(self.column_names)}). Vérifiez le pipeline.")

        df = pd.DataFrame(X, columns=self.column_names)

        return df

if __name__ == '__main__':
    # Exemples de test pour les transformeurs
    data = pd.DataFrame({
        'income': [50000.0, 60000.0, np.nan, 70000.0],
        'loan_amount': [10000.0, np.nan, 15000.0, 20000.0],
        'payment_delays': [0, 1, 3, np.nan],
        'employment_length': [5.0, 10.0, 2.0, np.nan],
        'age': [30, 40, 25, 50],
        'credit_score': [700, 750, 680, 720],
        'debt_to_income_ratio': [0.3, 0.25, 0.4, 0.35],
        'payment_history_score': [0.9, 0.85, 0.8, 0.95],
        'num_dependents': [2, 1, 3, 0],
        'employment_type': ['salaried', 'self-employed', 'salaried', 'salaried'],
        'marital_status': ['single', 'married', 'single', 'married'],
        'home_ownership': ['rent', 'own', 'rent', 'own'],
        'some_other_col': ['A', 'B', 'C', 'D'],
        'default': [0, 0, 1, 0]
    })

    X = data.drop(['default', 'some_other_col'], axis=1) # Tester uniquement les colonnes numériques et catégorielles
    preprocessor = build_preprocessing_pipeline(imputer_type='median')
    X_processed_array = preprocessor.fit_transform(X)

    print("\nOutput du ColumnTransformer (array numpy):")
    print(X_processed_array)
    print(f"Forme: {X_processed_array.shape}")


    # Tester ArrayToDataFrameTransformer
    array_to_df = ArrayToDataFrameTransformer()
    X_processed_df = array_to_df.transform(X_processed_array) # Note: fit() n'est pas nécessaire si column_names est fixe

    print("\nOutput de ArrayToDataFrameTransformer (DataFrame):")
    print(X_processed_df)
    print(f"Colonnes: {X_processed_df.columns.tolist()}")
    print(f"Forme: {X_processed_df.shape}")