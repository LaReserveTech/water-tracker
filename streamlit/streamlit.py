import streamlit as st
import pandas as pandas
import numpy as np
import pybase64 as base64
import seaborn as sns 
import matplotlib.pypot as plt
import plotly.express as px 

## Début du display
st.title('Water Tracker')

st.sidebar.title('Navigation')
st.sidebar.radio('Indicateurs de niveaux d'eau', [":rainbow["Sècheresse"]", "Autre indicateur" , "Water Tracker:movie_camera:"])
pages = ["Sècheresse", "Autre indicateur" , "Water Tracker"]


# connection = connection to db / addresse de files dans S3 (dispo sous url)
[gcp_service_account]
type = "service_account"
project_id = "xxx"
private_key_id = "xxx"
private_key = "xxx"
client_email = "xxx"
client_id = "xxx"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "xxx"

# streamlit_app.py

import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

# Perform query.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def run_query(query):
    query_job = client.query(query)
    rows_raw = query_job.result()
    # Convert to list of dicts. Required for st.cache_data to hash the return value.
    rows = [dict(row) for row in rows_raw]
    return rows

rows = run_query("SELECT word FROM `bigquery-public-data.samples.shakespeare` LIMIT 10")

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

  
