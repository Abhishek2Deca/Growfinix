import streamlit as st
import pandas as pd
import numpy as np
import joblib

kmeans = joblib.load('Kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Customer Segmentation App')

st.write('Enter customer details to predict the segment')

# NOTE: these ranges/defaults now match the scale of the data the model
# was actually trained on (raw dollar Income, raw dollar Total_Spend, etc.)
# Using tiny ranges like 0-100 for Income/Total_Spend caused every input
# to scale to nearly the same extreme z-score, collapsing all predictions
# into a single cluster.

age = st.number_input('Age', min_value=18, max_value=100, value=40)
income = st.number_input('Annual Income ($)', min_value=0, max_value=700000, value=52000, step=1000)
total_spend = st.number_input('Total Spend ($)', min_value=0, max_value=3000, value=600, step=10)
num_web_purchases = st.number_input('Number of Web Purchases', min_value=0, max_value=30, value=4)
num_store_purchases = st.number_input('Number of Store Purchases', min_value=0, max_value=15, value=5)
web_visits = st.number_input('Number of Web Visits (per month)', min_value=0, max_value=20, value=6)
recency = st.number_input('Recency (days since last purchase)', min_value=0, max_value=100, value=50)

input_data = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'Total_Spend': [total_spend],
    'NumWebPurchases': [num_web_purchases],
    'NumStorePurchases': [num_store_purchases],
    'NumWebVisitsMonth': [web_visits],
    'Recency': [recency]
})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.write(f'Predicted Segment: {cluster}')

    st.write("Raw input:", input_data)
    st.write("Scaled input:", input_scaled)