# Projet d'Évaluation du Risque de Crédit

## Description du Projet

Ce projet implémente un système d'évaluation automatisé du risque de crédit basé sur le machine learning. Il permet d'évaluer la probabilité qu'un emprunteur fasse défaut sur son prêt en se basant sur diverses caractéristiques financières et personnelles.

### Pourquoi c'est utile ?

1. **Pour les Institutions Financières** :
   - Automatisation de l'évaluation des risques
   - Standardisation du processus de décision
   - Réduction des coûts opérationnels
   - Amélioration de la gestion des risques

2. **Pour les Analystes de Crédit** :
   - Support à la décision basé sur des données
   - Traitement rapide des demandes
   - Évaluation objective et cohérente

3. **Pour les Clients** :
   - Décisions de crédit plus rapides
   - Processus transparent et équitable
   - Évaluation basée sur des critères objectifs

## Architecture du Projet

### 1. Structure des Données
Le modèle utilise 5 caractéristiques principales :
- `income` : Revenu annuel du client
- `loan_amount` : Montant du prêt demandé
- `payment_delays` : Nombre de retards de paiement antérieurs
- `employment_length` : Durée d'emploi en années
- `age` : Âge du client

### 2. Composants Principaux

#### Module de Génération de Données (`src/data_generation.py`)
- Génère des données synthétiques réalistes pour le développement et les tests
- Crée des relations cohérentes entre les variables
- Introduit des valeurs manquantes contrôlées (5-15%)

#### Pipeline de Prétraitement (`src/preprocessing.py`)
- Gestion des valeurs manquantes par imputation médiane
- Standardisation des variables numériques
- Conversion des formats de données

#### Ingénierie des Caractéristiques (`src/features.py`)
- Calcul du ratio dette/revenu
- Score de risque basé sur l'âge
- Score d'historique de paiement
- Génération de caractéristiques composites

#### Entraînement du Modèle (`src/model_training.py`)
- Utilise GradientBoostingClassifier
- Optimisation des hyperparamètres via GridSearchCV
- Validation croisée en 3 plis
- Métriques d'évaluation : ROC-AUC, précision, rappel

#### Service de Prédiction (`src/prediction_service.py`)
- Chargement et gestion du modèle entraîné
- Interface unifiée pour les prédictions
- Validation des données d'entrée
- Gestion des erreurs

#### API REST (`api/app.py`)
- Endpoints :
  - GET / : Vérification de l'état du service
  - POST /predict : Prédictions individuelles ou en lot
- Format JSON pour les requêtes/réponses
- Gestion des erreurs HTTP

### 3. Pipeline Machine Learning
1. **Prétraitement** :
   - Imputation des valeurs manquantes
   - Standardisation des variables
2. **Feature Engineering** :
   - Création de nouvelles caractéristiques
   - Transformations des variables
3. **Modélisation** :
   - GradientBoostingClassifier optimisé
   - Probabilités de défaut calibrées

## Performance du Modèle

- **ROC-AUC** : ~0.86 (validation)
- **Précision** : ~0.71
- **Rappel** : ~0.54

Ces métriques indiquent :
- Bonne capacité de discrimination globale
- Précision élevée dans les prédictions positives
- Compromis conservateur sur le rappel

## Installation et Utilisation

### Prérequis
- Python 3.13+
- Poetry pour la gestion des dépendances

### Installation
```bash
# Cloner le repository
git clone [url-du-repo]
cd credit_risk_project

# Installer les dépendances avec Poetry
poetry install
```

### Utilisation

1. **Générer les données** (si nécessaire) :
```bash
poetry run python src/data_generation.py
```

2. **Entraîner le modèle** :
```bash
poetry run python src/model_training.py
```

3. **Lancer l'API** :
```bash
poetry run python api/app.py
```

### Exemple d'Utilisation de l'API

```python
import requests

# Données d'un client
client_data = {
    "income": 75000.0,
    "loan_amount": 30000.0,
    "payment_delays": 1,
    "employment_length": 10.0,
    "age": 45
}

# Requête à l'API
response = requests.post('http://localhost:5000/predict', json=client_data)
probability = response.json()['probability_of_default']
```

## Déploiement

Le projet inclut :
- Dockerfile pour la containerisation
- Charts Helm pour le déploiement Kubernetes
- Configuration de l'environnement de production

## Améliorations Futures Possibles

1. **Modélisation** :
   - Essai d'autres algorithmes (XGBoost, LightGBM)
   - Techniques avancées pour le déséquilibre des classes
   - Optimisation plus poussée des hyperparamètres

2. **Données** :
   - Intégration de sources de données réelles
   - Ajout de variables catégorielles
   - Enrichissement des données externes

3. **API** :
   - Authentication et autorisation
   - Rate limiting
   - Versioning des modèles
   - Monitoring des performances

4. **Infrastructure** :
   - Mise en place de tests de charge
   - Monitoring des prédictions
   - Pipeline CI/CD complet

## Sécurité et Conformité

- Validation des données d'entrée
- Logging des prédictions
- Pas de stockage de données sensibles
- Conformité RGPD (pour les données réelles)