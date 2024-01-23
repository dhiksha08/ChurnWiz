import streamlit as st

# Set Streamlit theme to dark, enable "Run on Save," and set default layout to wide

st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")
st.markdown("<h1 style='text-align: center; color : #E8570E; font-size:45px'>Telecom Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<br><hr><br>", unsafe_allow_html=True)

st.write(
        """
        ## Overview

        Welcome to the Telecom Churn Prediction project! Churn rate is the percentage of subscribers to a service that discontinue their subscription to that service in a given time period. With growing pressure from competition and government mandates improving retention rates of profitable customers has become an increasingly urgent to telecom service providers.
        This project analyses the factors affecting telecom customer churn and models a predictor for telecom customer churn.

        ## Dataset

        The dataset used for this project contains a rich set of features, Data fields including state,account length,area code,voicemail plan,total_intl_calls,total_intl_charge,number of customer service_calls,churn,and more. Exploring this dataset provides valuable insights into the patterns and factors influencing cutsomer churn.

        ## Approach

        Our approach for the telecom churn prediction project involves a combination of machine learning algorithms, such as Logistic Regression, k-Nearest Neighbors (KNN), XGBoost, Gradient Boosting, and Random Forest. By training the model on historical customer data, we aim to capture the complex patterns and relationships between various features and customer churn behavior, enabling accurate predictions for identifying potential churners in the future.
        
        ## Key Features

        - **Among the models trained, XGBoost stands out as the best algorithm,delivering the most accurate predictions.

        ## Usage

        1. **Data Preprocessing:** Before running the models, preprocess the dataset to handle missing values, convert imbalanced data to balanced data, and normalize features.
        2. **Model Training:** Train the models using the processed dataset.
        3. **Prediction:** Trained model to make predictions on telecom churn data.

        ## Contributors
        - Abinaya V  - 21pd01
        - Dhikshitha - 21pd26
        - Shivani E  - 21pd34

        """
    )