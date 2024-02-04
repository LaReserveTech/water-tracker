import streamlit as st
import pandas as pandas
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px 

st.sidebar.title('Navigation')
pages = [":rainbow: Sècheresse", "Autre indicateur", 'FAQ']
page = st.sidebar.radio('Water Tracker', pages)

## Début du display
if page == pages[0]:
    st.title('Water Tracker')
    st.header('Statistiques et visualisations des données sécheresse en France métropolitaine en 2023')
    st.write('Water Tracker est un outil permettant de suivre l’évolution de la sécheresse et ses impacts, en France métropolitaine sur l’année 2023.')

    st.title('ÉVOLUTION DE LA SÉCHERESSE EN 2023')
    st.markdown('''Un épisode de sécheresse peut survenir si nos ressources en eau sont en tension ou ont été en tension pendant une trop longue période. 
Nous disposons de 3 ressources en eau principales :

- La pluie
- Les nappes phréatiques
- Les eaux de surface (fleuves, rivières, lacs)

Pour plus d’informations sur la sécheresse [rdv 
ici](https://www.ecologie.gouv.fr/secheresse).''')



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
    
