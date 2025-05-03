import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DebtToIncomeTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer personnalisé pour calculer le ratio Dette sur Revenu.
    Gère les cas de revenu nul ou très faible après scaling.
    """
    def __init__(self, income_col='income', loan_amount_col='loan_amount'):
        self.income_col = income_col
        self.loan_amount_col = loan_amount_col

    def fit(self, X, y=None):
        """
        Rien à apprendre ici pour ce transformeur simple.
        """
        # Calculer la médiane du ratio pour gérer les cas inf/NaN si besoin (pour robustesse)
        # Attention: Ceci devrait idéalement être fait sur les données d'entraînement SEULEMENT.
        # Si appelé après scaling, les valeurs sont différentes.
        # Dans le pipeline, ce transformeur sera appelé APRES le scaling.
        # Nous calculons donc la médiane sur les données scalées dans la phase de FIT
        # (qui ne fait rien ici), mais l'application réelle (remplacement des inf/nan)
        # se fait dans TRANSFORM, potentiellement en utilisant une statistique calculée ailleurs
        # (comme la médiane du rapport *après* scaling sur le train set, qui n'est pas stockée ici).
        # Pour simplifier ici, on gère inf/NaN en remplaçant par la médiane des VALEURS CALCULÉES
        # dans le transform, ce qui est moins idéal car elle dépend du batch.
        # Une approche robuste stockerait la médiane du rapport calculée sur le set d'entraînement.
        # Simplification: On remplace simplement les inf/nan par une constante raisonnable (0 ou 1, ou médiane du batch transformé)
        # Utilisons la médiane du batch transformé pour cet exemple simple.
        return self

    def transform(self, X):
        """
        Applique la transformation.
        """
        X_transformed = X.copy()

        # Assurez-vous que les colonnes existent (elles devraient si le preprocessing les a laissées passer)
        if self.income_col not in X_transformed.columns or self.loan_amount_col not in X_transformed.columns:
            raise ValueError(f"Colonnes requises ({self.income_col}, {self.loan_amount_col}) non trouvées.")

        # IMPORTANT: Gérer la division par zéro ou valeurs très proches de zéro APRES SCALING.
        # Ajouter une petite constante pour éviter la division par zéro exacte.
        # Après scaling, income peut être négatif ou proche de zéro.
        # Une division par une valeur proche de zéro ou négative peut donner des résultats extrêmes.
        # La logique dépend de l'interprétation après scaling.
        # Ici, on traite les valeurs "proches de zéro après scaling" comme posant problème.
        # Un petit epsilon ajouté au dénominateur gère la division par zéro exacte.
        # Remplacer les infinis et NaN résultants.

        # Calcul du rapport (peut générer Inf ou NaN si income est 0 ou NaN)
        # La gestion des NaN initiaux doit être faite AVANT ce transformer (par l'imputer).
        # Donc ici, income et loan_amount ne devraient PAS avoir de NaN (sauf si pas numériques).
        # income_col_values = X_transformed[self.income_col].values # Utilisez .values pour performance
        # loan_amount_col_values = X_transformed[self.loan_amount_col].values
        # ratio = loan_amount_col_values / (income_col_values + 1e-6) # Ajoutez epsilon
        # Alternative: Use pandas Series operations directly which are safer.
        ratio = X_transformed[self.loan_amount_col] / (X_transformed[self.income_col] + 1e-6)


        # Remplacer les infinis et NaN résultants par une valeur raisonnable (ex: la médiane du ratio calculé)
        # Calculer la médiane *après* avoir géré la division par zéro pour ne pas inclure les inf/NaN dans la médiane.
        # Les valeurs inf/NaN sont généralement dues à la division par (proche de) zéro après scaling.
        # Exclure infinis et NaNs pour calculer la médiane
        finite_ratios = ratio[np.isfinite(ratio)]
        median_ratio = finite_ratios.median() if not finite_ratios.empty else 0.0 # Gérer le cas où toutes les valeurs sont inf/nan

        ratio = ratio.replace([np.inf, -np.inf], np.nan) # Replace inf with NaN first
        ratio = ratio.fillna(median_ratio)             # Puis replace NaN avec la médiane

        X_transformed['debt_to_income'] = ratio

        return X_transformed

class AgeRiskTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer personnalisé pour créer un facteur de risque basé sur l'âge.
    """
    def __init__(self, age_col='age', optimal_age=40):
        self.age_col = age_col
        self.optimal_age = optimal_age

    def fit(self, X, y=None):
        """
        Rien à apprendre ici.
        """
        return self

    def transform(self, X):
        """
        Applique la transformation.
        """
        X_transformed = X.copy()
         # Assurez-vous que les colonnes existent
        if self.age_col not in X_transformed.columns:
             raise ValueError(f"Colonne requise ({self.age_col}) non trouvée.")

        # Calculer la distance par rapport à l'âge optimal (valeur absolue)
        # Gérer les NaN potentiels dans la colonne age si l'imputer n'a pas été appliqué correctement avant
        # (Ce transformeur doit être APRES l'imputation/scaling de 'age' dans le pipeline)
        # Donc 'age' ne devrait pas avoir de NaN ici.
        X_transformed['age_risk_factor'] = np.abs(X_transformed[self.age_col] - self.optimal_age)

        return X_transformed

class PaymentHistoryScoreTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer personnalisé pour créer un score d'historique de paiement.
    Basé sur payment_delays. Plus de retards -> score plus bas.
    Gère les valeurs potentiellement négatives après scaling.
    """
    def __init__(self, delays_col='payment_delays'):
        self.delays_col = delays_col

    def fit(self, X, y=None):
        """
        Rien à apprendre ici.
        """
        return self

    def transform(self, X):
        """
        Applique la transformation.
        """
        X_transformed = X.copy()
        # Assurez-vous que les colonnes existent
        if self.delays_col not in X_transformed.columns:
             raise ValueError(f"Colonne requise ({self.delays_col}) non trouvée.")

        # Calculer un score simple : 1 / (1 + |retards|)
        # Utiliser la valeur absolue pour gérer les valeurs négatives après scaling.
        # Gérer les NaN potentiels dans delays_col (doit être géré par l'imputer avant)
        # Donc delays_col ne devrait pas avoir de NaN ici.
        X_transformed['payment_history_score'] = 1 / (1 + np.abs(X_transformed[self.delays_col]))

        return X_transformed

# Utilitaires pour combiner les transformations en un seul pas pour le pipeline
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer combiné pour appliquer toutes les transformations d'ingénierie des caractéristiques.
    """
    def __init__(self):
        # Initialiser les transformeurs que nous allons appliquer
        self.dti_transformer = DebtToIncomeTransformer()
        self.age_transformer = AgeRiskTransformer()
        self.payment_transformer = PaymentHistoryScoreTransformer()

    def fit(self, X, y=None):
        """
        Ajuster les transformeurs internes si nécessaire (aucun ajustement dans nos exemples simples).
        """
        # Aucun de nos transformeurs actuels ne nécessite un fit.
        # S'ils en nécessitaient un (ex: calcul d'une statistique sur le set d'entraînement),
        # on appellerait self.transformer.fit(X, y).
        self.dti_transformer.fit(X, y) # Appel factice pour la complétude
        self.age_transformer.fit(X, y)
        self.payment_transformer.fit(X, y)
        return self

    def transform(self, X):
        """
        Appliquer séquentiellement toutes les transformations.
        """
        X_transformed = X.copy()

        # Appliquer chaque transformeur
        X_transformed = self.dti_transformer.transform(X_transformed)
        X_transformed = self.age_transformer.transform(X_transformed)
        X_transformed = self.payment_transformer.transform(X_transformed)

        return X_transformed

if __name__ == '__main__':
     # Exemple d'utilisation du transformeur d'ingénierie des caractéristiques
    data_processed = pd.DataFrame({
        'income': [0.5, 1.0, -0.2, 1.5], # Exemples de valeurs après scaling
        'loan_amount': [0.1, -0.5, 0.3, 0.8],
        'payment_delays': [-1.0, -0.5, 0.8, 0.1],
        'employment_length': [0.2, 1.2, -0.8, 0.5],
        'age': [-1.5, 0.1, -2.0, 0.8],
        'col_remainder': ['A', 'B', 'C', 'D'] # Colonne passée par remainder
    })

    fe_transformer = FeatureEngineeringTransformer()

    # Normalement, fit_transform serait appelé sur les données d'entraînement
    # Pour montrer l'effet, on l'appelle ici directement (en ignorant le fit qui ne fait rien)
    data_fe = fe_transformer.transform(data_processed)

    print("\nDonnées après ingénierie des caractéristiques:")
    print(data_fe)
    print(f"Colonnes ajoutées: {['debt_to_income', 'age_risk_factor', 'payment_history_score']}")
    print(f"Forme: {data_fe.shape}")
