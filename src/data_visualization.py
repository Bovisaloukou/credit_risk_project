import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_credit_data(csv_path='data/generated_credit_data.csv', output_dir='data/visualizations'):
    """
    Génère des visualisations pour explorer les distributions et corrélations des variables du dataset de crédit.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data = pd.read_csv(csv_path)
    # Histogrammes pour toutes les variables numériques
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(data[col].dropna(), kde=True, bins=30)
        plt.title(f'Distribution de {col}')
        plt.savefig(f'{output_dir}/hist_{col}.png')
        plt.close()
    # Heatmap de corrélation
    plt.figure(figsize=(10, 8))
    corr = data[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Matrice de corrélation')
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()
    # Boxplots pour variables catégorielles vs. défaut
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=col, y='credit_score', data=data)
        plt.title(f'Credit score par {col}')
        plt.savefig(f'{output_dir}/boxplot_credit_score_{col}.png')
        plt.close()
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=col, y='debt_to_income_ratio', data=data)
        plt.title(f'Debt-to-income ratio par {col}')
        plt.savefig(f'{output_dir}/boxplot_dti_{col}.png')
        plt.close()
    # Barplot du taux de défaut par catégorie
    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        default_rate = data.groupby(col)['default'].mean()
        sns.barplot(x=default_rate.index, y=default_rate.values)
        plt.title(f'Taux de défaut par {col}')
        plt.ylabel('Taux de défaut')
        plt.savefig(f'{output_dir}/default_rate_{col}.png')
        plt.close()
    print(f"Visualisations sauvegardées dans {output_dir}")

if __name__ == "__main__":
    visualize_credit_data()
