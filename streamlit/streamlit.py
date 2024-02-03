import streamlit as st
import pandas as pandas
import numpy as np
import pybase64 as base64
import seaborn as sns 
import matplotlib.pypot as plt
import plotly.express as px 

## Début du display
st.title('Water Tracker')
st.header('Statistiques et visualisations des données sécheresse en France métropolitaine en 2023')
st.write('Water Tracker est un outil permettant de suivre l’évolution de la sécheresse et ses impacts, en France métropolitaine sur l’année 2023.')

st.title('ÉVOLUTION DE LA SÉCHERESSE EN 2023')
st.write('Un épisode de sécheresse peut survenir si nos ressources en eau sont en tension ou ont été en tension pendant une trop longue période. Nous disposons de 3 ressources en eau principales :

- La pluie
- Les nappes phréatiques
- Les eaux de surface (fleuves, rivières, lacs)

Pour plus d’informations sur la sécheresse [rdv ici](%s)'%(https://www.ecologie.gouv.fr/secheresse).)

st.sidebar.title('Navigation')
st.sidebar.radio('Indicateurs de niveaux d'eau', [":rainbow["Sècheresse"]", "Autre indicateur" , "Water Tracker:movie_camera:"])
pages = ["Sècheresse", "Autre indicateur" , "Water Tracker"]


# Print results.
st.write("Some wise words from Shakespeare:")
for row in rows:
    st.write("✍️ " + row['word'])

#df = to_dataframe(data extracted from database) 

if page == pages[0]:
  #things to put on the page
  fig, ax = plt.subplots()
  ax.plot(df.x, df.y)
  st.title('title')
  st.pyplot(fig)
if page == pages[1]:
  #things to put on the second page
if page == pages[2]:
  #things to put on the third page

  
