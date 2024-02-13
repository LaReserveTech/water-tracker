import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plotly.express as px 

st.sidebar.title('Navigation')
pages = [":rainbow: Evolution de la sècheresse", "Impacts de la sècheresse", 'FAQ']
page = st.sidebar.radio('Water Tracker', pages)

#data
df_precipitations = pd.read_csv('data/pluviométrie_data.csv')
df_flow = pd.read_csv('data/df_stations.csv')
df_nappes = pd.read_csv('data/nappes_data.csv')

#Plot1
@st.cache_data
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
@st.cache_data
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

@st.cache_data
def plot_precipitations(df):
    #Plot3

    # Couleurs
    #J'aurais mis les couleurs dans l'autre sens
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


## Début du display
if page == pages[0]:
    st.title('Water Tracker')

# C'est quoi les conséquences de la sècheresse
    
    st.header('Statistiques et visualisations des données sécheresse en France métropolitaine en 2023')
    st.markdown('<div style="text-align: justify;">Water Tracker est un outil permettant de suivre l’évolution de la sécheresse et ses impacts, en France métropolitaine sur l’année 2023.</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify;"><br></br></div>', unsafe_allow_html=True)

#Il faudrait une photo, une image, un divider ou changer le display avec le titre ou la page (ou alors enlever Water Tracker et ne mettre qu'un titre en capitales)

    st.header('Évolution de la sècheresse en 2023')
    st.markdown('''Un épisode de sécheresse peut survenir si nos ressources en eau sont en tension ou ont été en tension pendant une trop longue période. 
Nous disposons de 3 ressources en eau principales :

- La pluie
- Les nappes phréatiques
- Les eaux de surface (fleuves, rivières, lacs)

Pour plus d’informations sur la sécheresse [rdv 
ici](https://www.ecologie.gouv.fr/secheresse).''')
    
    st.markdown('<div style="text-align: justify;"><br></br></div>', unsafe_allow_html=True)
    
    st.header(":umbrella_with_rain_drops: ÉVOLUTION DE LA PLUIE EN 2023", divider='rainbow')
    st.markdown('<div style="text-align: justify;">La pluie apporte une grande quantité d’eau sur le territoire. Deux-tiers des volumes précipités s’évaporent et le reste vient alimenter la végétation et nos réserves d’eau (nappes et eaux de surface). Des précipitations insuffisantes ont donc un impact important sur la résilience du territoire. <br></br></div>', unsafe_allow_html=True)
    
    fig = plot_precipitations(df_precipitations)
    st.pyplot(fig)

    st.markdown('<div style="text-align: justify;">Comment lire le graphique:<br></br> En novembre 2023, 35% des stations de relevé indiquaient un niveau de précipitation très bas à modérément bas; 10% indiquaient un niveau comparable aux normales de saison; 55% indiquaient un niveau modérément haut à très haut.<br></br><br></br></div>', unsafe_allow_html=True)

#Ca a quoi comme impact ? Et surtout c'est quoi niveau de sècheresse ? S'agit-il des précipitations ou d'un indicateur de précipitations / commentaires sur novembre et décembre sur tous les graphes

    st.header(':sailboat:ÉVOLUTION DU TAUX DE REMPLISSAGE DES EAUX DE SURFACE EN 2023', divider='rainbow')
    st.markdown('<div style="text-align: justify;">Les eaux de surface (fleuves, rivières, lacs) constituent la grande majorité de nos réserves en eau (plus de 80%). On vient régulièrement y puiser pour alimenter les villes et les industries. Elles sont alimentées principalement par la pluie et les nappes.<br></br></div>', unsafe_allow_html= True)

    fig = plot_flow(df_flow)
    st.pyplot(fig)

    # J'aurais mis les couleurs dans l'autre sens, c'est le débit de quoi ? 

    st.markdown('<div style="text-align: justify;">Comment lire le graphique:<br></br> En novembre 2023, 35% des eaux de surfaces avaient un niveau très bas à modérément bas; 10% avaient un niveau comparable aux normales de saison; 55% avaient un niveau modérément haut à très haut.<br></br><br></br></div>', unsafe_allow_html=True)

    st.header(":bucket: ÉVOLUTION DU TAUX DE REMPLISSAGE DES NAPPES PHRÉATIQUES EN 2023", divider='rainbow')
    st.markdown('<div style="text-align: justify;">Les nappes constituent une autre recharge d’eau cruciale pour alimenter nos besoins en eau toute l’année. En principe, elles se remplissent pendant les mois d’automne et d’hiver grâce à la pluie. Mais leur taux de remplissage peut être inférieur aux normales de saison s’il ne pleut pas assez et menacer nos stocks pour faire face aux mois plus chauds.<br></br></div>', unsafe_allow_html=True)

    fig = plot_groundwater(df_nappes)
    st.pyplot(fig)

    st.markdown('<div style="text-align: justify;">Comment lire le graphique:<br></br> En novembre 2023, 35% des nappes avaient un niveau très bas à modérément bas; 10% avaient un niveau comparable aux normales de saison; 55% avaient un niveau modérément haut à très haut.</div>', unsafe_allow_html=True)

#Il aut vérifier les chiffres

if page == pages[1]:
    st.title('Water Tracker')
    st.header('LES IMPACTS DE LA SÉCHERESSE EN 2023')  
    st.write('L’une des manières de voir l’impact de la sécheresse sur la biodiversité est de regarder l’évolution de la qualité des cours d’eau (en termes de caractéristiques bio-physiques et en termes de températures). La sécheresse augmente la concentration de polluants et la température des cours d’eau, ce qui peut durablement impacter les écosystèmes. Le taux de remplissage des nappes est le meilleur moyen de rendre compte de la sécheresse sur la qualité des cours d’eau car XXX')

#Remplacer XXX et qu'est-ce que ça a comme impact quand on passe en dessous de tel ou tel pourcentage, pour l'écosystème, pour l'approvisionnement, etc.

    st.markdown('<div style="text-align: justify;">Comment lire le graphique:<br></br> En octobre 2023, 85% des stations d’analyse rendaient compte d’une qualité des cours d’eau mauvaise ou médiocre. En même temps, 60% des nappes avaient un niveau en dessous des normales de saison.<br></br></div>', unsafe_allow_html=True)
    st.header('&#10024; LA SECHERESSE ET LES RESTRICTIONS D’EAU')
    st.write('Des restrictions préfectorales sont parfois mises en place lorsque nos ressources en eau sont en tension. Visitez [vigieau.gouv.fr](http://vigieau.gouv.fr) pour savoir si vous êtes concerné.e par une restriction d’eau aujourd’hui.')

if page == pages[2]:
    st.title('FAQ')
    

    if st.button("- Y a-t-il différents types de sécheresse ?"):
        st.markdown('''La sécheresse est un phénomène compliqué et multi-factoriel. On en 
distingue 3 grands types :

- La sécheresse météorologique, provoquée par un manque de pluie
- La sécheresse agricole, causée par un manque d’eau dans les sols, ce 
qui altère le développement de la végétation
- La sécheresse hydrologique lorsque les lacs, rivières, cours d’eau 
ou nappes souterraines ont des niveaux anormalement bas
    
    ([Source](https://www.ecologie.gouv.fr/secheresse))''')
        
    if st.button('- Une sécheresse peut-elle en entraîner une autre ?'):
        st.markdown('''Oui ! S’il ne pleut pas pendant une période anormalement longue, 
une sécheresse météorologique peut se déclencher. Si cela perdure, les 
sols s’assèchent, ce qui crée une sécheresse agricole. Les stocks d’eau, 
c’est-à-dire des nappes phréatiques, des barrages et des cours d’eau, ne 
sont plus alimentés. Leur niveau commence à baisser, ce qui peut entraîner 
une sécheresse hydrologique. 
        
Le nombre de jours sans pluie conduisant à une sécheresse agricole ou 
hydrologique change considérablement en fonction du climat et de la 
saison, de la typologie du sol et de la végétation existante. En règle 
générale, cette période est significativement plus courte en pleine été 
qu'au début du printemps.
    
Le déclenchement d'une sécheresse est également influencé par les 
saisons antérieures. Une insuffisance de recharge hivernale accroît 
grandement la probabilité d'une sécheresse durant l'été qui suit. 
D'ailleurs, les périodes de sécheresse sévère découlent souvent de 
déficits pluviométriques répétés sur plusieurs saisons d'affilée.
    
    
([Source](https://www.eaufrance.fr/la-secheresse#:~:text=La%20s%C3%A9cheresse%20m%C3%A9t%C3%A9orologique%20correspond%20%C3%A0,%C3%A9daphique%20(ou%20s%C3%A9cheresse%20agricole)))''')

    
    if st.button('- Quel est l’impact du réchauffement climatique sur la sécheresse?'):
        st.markdown('''Le changement climatique rendent les sécheresses plus 
fréquentes. Celles-ci arrivent aussi plus tôt dans l’année. La hausse des 
températures, en particulier, augmente l’évaporation et réduit donc le 
remplissage de nos réserves d’eau. Les effets de la sécheresse sont déjà 
visibles, notamment en Méditerranée. ''')
    
    if st.button('- Comment protéger nos ressources en eau ?'):
        st.markdown('''Chacun peut agir. 
    
Les citoyens peuvent diminuer leur consommation en adoptant des gestes 
simples tels que préférer les douches aux bains, installer des toilettes à 
double chasse et des réducteurs de pression sur les robinets, ou encore en 
récupérant l'eau de pluie.
    
Parallèlement, les entreprises et les agriculteurs sont incités à 
adopter des pratiques plus économes en eau : optimiser leurs procédés pour 
réduire la consommation d'eau, utiliser des systèmes en circuit fermé, et 
recycler les eaux utilisées pour le nettoyage.
    
Malgré tout, les préfets doivent parfois prendre des mesures plus 
fortes et décreter des restrictions d’eau pour éviter les sécheresses ou 
réduire leur impact. ''')
        
    if st.button('- Comment a été construit ce site ?'):
        st.markdown('''Ce site a été créé par LaReserve.tech, le programme de 
mobilisation citoyenne pour créer des réponses rapides aux urgences 
sociales et environnementales. Une équipe d’une dizaine de bénévoles ont 
compilé des données publiques en lien avec la sécheresse pour offrir une 
vision simplifiée de ce phénomène complexe, au bénéfice de tous. La 
Réserve est un programme opéré par l’ONG Bayes Impact.''')
    
