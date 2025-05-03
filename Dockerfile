# Utiliser une image Python officielle comme base
FROM python:3.9-slim as base

# Installer Poetry
# Utiliser la méthode recommandée par Poetry
ENV POETRY_VERSION=1.1.13 # Spécifier la version ou utiliser une version récente
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_IN_PROJECT=false # Ne pas créer l'env virtuel dans /app/.venv
ENV POETRY_NO_INTERACTION=1 # Désactiver les prompts interactifs

RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="$POETRY_HOME/bin:$PATH" # Ajouter poetry au PATH

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de configuration de Poetry
# Copier pyproject.toml et poetry.lock pour l'installation déterministe des dépendances
COPY pyproject.toml poetry.lock* ./

# Installer les dépendances du projet à l'aide de Poetry
# Utiliser l'option --no-dev pour ne pas installer les dépendances de développement (comme pytest, si ajoutées)
RUN poetry install --no-dev --no-root

# Copier le code de l'application
COPY src/ ./src/
COPY api/ ./api/

# Copier le modèle entraîné
COPY models/ ./models/

# Exposer le port
EXPOSE 5000

# Commande pour lancer l'application avec Gunicorn via Poetry
# Utilisez 'poetry run' pour vous assurer que l'environnement de Poetry est utilisé
CMD ["poetry", "run", "gunicorn", "-b", "0.0.0.0:5000", "api.app:app"]

# Alternative simple pour le développement ou si poetry run pose problème avec gunicorn:
# Identifier le chemin de l'exécutable gunicorn dans l'env virtuel de poetry
# RUN poetry run which gunicorn # Utile pour debug, trouver le path
# CMD ["/root/.local/share/pypoetry/virtualenvs/mon-projet-XXXX/bin/gunicorn", ...] # Exemple de path (peut varier)
# MAIS `poetry run` est la méthode standard et préférée.