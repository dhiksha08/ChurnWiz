import pandas as pd
import streamlit as st
import numpy as np





st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")
st.markdown("<h1 style='text-align: center; color : #E8570E; font-size:65px'>Telecom Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(""" ### Model Selection """)
st.write(f"After careful evaluation of different machine learning classifiers, the **XGBoost** stands out as the top performer based on the evaluation metrics. Let's take a look at its performance across various metrics:")
 

model_metrics = pd.read_csv("Models/model_metrics.csv")

st.table(model_metrics)
