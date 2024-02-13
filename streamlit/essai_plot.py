import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

#Plot1
def plot_flow(df):

    # Couleurs
    colors = ["#da442c", "#f28f00", "#ffdd55", "#6cc35a", "#30aadd", "#1e73c3", "#286172"]

    # Tracé
    ax = df.plot(kind='bar', stacked=True, color=colors, figsize=(10, 6))

    # Ajustements esthétiques
    plt.yticks(range(0, 101, 10))
    plt.title('Débit: Répartition des niveaux de sécheresse en 2023')
    plt.xlabel('')
    plt.ylabel('Proportion des stations (%)')

    # Utiliser les noms de mois traduits sur l'axe des abscisses
    ax.set_xticklabels([date for date in df['Mois']], rotation=45, ha='right')

    # Légende
    legend_labels = ["Très bas", "Bas", "Modérément bas", "Autour de la normale", "Modérément haut", "Haut", "Très haut"]
    legend_colors = colors[::-1]  # Inverser l'ordre des couleurs

    handles = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.0, 0.5), title='Niveaux de Sécheresse')

    # Réglage automatique de l'orientation des dates sur l'axe des x
    fig = ax.get_figure()
    fig.autofmt_xdate(rotation=45)
    return fig


#Plot2
def plot_groundwater(df):

    # Couleurs
    colors = ["#da442c", "#f28f00", "#ffdd55", "#6cc35a", "#30aadd", "#1e73c3", "#286172"]

    # Tracé
    ax = df.plot(kind='bar', stacked=True, color=colors, figsize=(10, 6))

    # Ajustements esthétiques
    plt.yticks(range(0, 101, 10))
    plt.title('Nappes: Répartition des niveaux de sécheresse en 2023')
    plt.xlabel('')
    plt.ylabel('Proportion des stations (%)')

    # Utiliser les noms de mois traduits sur l'axe des abscisses
    ax.set_xticklabels([date for date in df['Mois']], rotation=45, ha='right')

    # Légende
    legend_labels = ["Très bas", "Bas", "Modérément bas", "Autour de la normale", "Modérément haut", "Haut", "Très haut"]
    legend_colors = colors[::-1]  # Inverser l'ordre des couleurs

    handles = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.0, 0.5), title='Niveaux de Sécheresse')

    # Réglage automatique de l'orientation des dates sur l'axe des x
    fig = ax.get_figure()
    fig.autofmt_xdate(rotation=45)
    return fig


def plot_precipitations(df):
    #Plot3
    # Charger le DataFrame à partir du fichier CSV
    df = pd.read_csv('./data/pluie_data.csv')

    # Couleurs
    colors = ["#da442c", "#f28f00", "#ffdd55", "#6cc35a", "#30aadd", "#1e73c3", "#286172"]

    # Tracé
    ax = df.plot(kind='bar', stacked=True, color=colors, figsize=(10, 6))

    # Ajustements esthétiques
    plt.yticks(range(0, 101, 10))
    plt.title('Pluie: Répartition des niveaux de sécheresse en 2023')
    plt.xlabel('')
    plt.ylabel('Proportion des stations (%)')

    # Utiliser les noms de mois traduits sur l'axe des abscisses
    ax.set_xticklabels([date for date in df['Mois']], rotation=45, ha='right')

    # Légende
    legend_labels = ["Sécheresse extrême", "Grande sécheresse", "Sécheresse modérée","Situation normale", "Modérément humide", "Très humide", "Extrêmement humide"]
    legend_colors = colors[::-1]  # Inverser l'ordre des couleurs

    handles = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.0, 0.5), title='Niveaux de Sécheresse')

    # Réglage automatique de l'orientation des dates sur l'axe des x
    fig = ax.get_figure()
    fig.autofmt_xdate(rotation=45)
    return fig

def main():
    
    essai_plot()

if __name__ == "__main__":
    main()