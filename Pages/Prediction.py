import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import scipy.stats as stats
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import plotly.offline as py
from collections import Counter
import plotly.graph_objects as go
import seaborn.objects as so
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import accuracy_score
import pickle
import os
from sklearn.model_selection import KFold
import multiprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier




st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")
st.markdown("<h1 style='text-align: center; color : #E8570E; font-size:65px'>Telecom Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

train=pd.read_csv("Dataset/train.csv")
test=pd.read_csv("Dataset/test.csv")


def replace_outliers(df, cols):
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3-q1

        lower_bound = q1-(1.5*iqr)
        upper_bound = q3+(1.5*iqr)

        median = df[col].median()

        df[col] = np.where(df[col]<lower_bound, median, df[col])
        df[col] = np.where(df[col]>upper_bound, median, df[col])

num_cols = train.select_dtypes(include=['int','float']).columns
replace_outliers(train,num_cols)

for data in [train, test]:
    data.drop(columns=['total_day_charge',
                       'total_eve_charge',
                      'total_night_charge',
                      'total_intl_charge'], inplace=True)

state_encoder = LabelEncoder()
area_code_encoder = LabelEncoder()
international_plan_encoder = LabelEncoder()
voice_mail_plan_encoder = LabelEncoder()

# Fit and transform the categorical columns in the training data
train['state'] = state_encoder.fit_transform(train['state'])
train['area_code'] = area_code_encoder.fit_transform(train['area_code'])
train['international_plan'] = international_plan_encoder.fit_transform(train['international_plan'])
train['voice_mail_plan'] = voice_mail_plan_encoder.fit_transform(train['voice_mail_plan'])


X = train.drop(columns='churn', axis=1)
Y = train['churn']


smote = SMOTE(random_state=42)
x_smote, y_smote = smote.fit_resample(X, Y)

oversampled_data = pd.DataFrame(data=x_smote,columns=x_smote.columns)
oversampled_data['churn']=y_smote
oversampled_data =oversampled_data.sample(frac=1)

x=oversampled_data.drop(['churn'],axis=1)
y=oversampled_data['churn']

# List of float numerical columns
float_numerical_columns = [
    'account_length', 'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
    'total_eve_minutes', 'total_eve_calls', 'total_night_minutes', 'total_night_calls',
    'total_intl_minutes', 'total_intl_calls', 'number_customer_service_calls'
]

# Create a StandardScaler object
scaler = StandardScaler()

# Apply standard scaling to the float numerical columns in X
x[float_numerical_columns] = scaler.fit_transform(x[float_numerical_columns])

# Now, X contains the scaled float numerical features
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


st.write("""
             
             ## Predict Telecom churn

            Welcome to the Predict Telecom churn page! Use the form on the left to input details for a customer churn, 
            and our trained model will predict the customer churn (yes/no).This page allows you to interactively explore 
            how the model performs on custom input data.""")



area_codes = ['area_code_415', 'area_code_408', 'area_code_510']
states = ['WV', 'MN', 'ID', 'AL', 'VA', 'OR', 'TX', 'UT', 'NJ', 'NY', 'OH', 'WY', 'WI', 'MA', 'ME', 'CT', 'RI', 
          'KS', 'MI', 'VT', 'MD', 'KY', 'NV', 'IN', 'MS', 'MO', 'NC', 'DE', 'MT', 'WA', 'CO', 'TN', 'IL', 'OK', 
          'NH', 'NM', 'HI', 'AZ', 'FL', 'SD', 'NE', 'SC', 'DC', 'AR', 'LA', 'PA', 'ND', 'GA', 'IA', 'AK', 'CA']



# Input form with appropriate spacing and alignment
col1, col2 = st.columns(2)
state = col1.selectbox("Select State", states)
account_length = col2.number_input("Account Length", min_value=0, format="%d")

col3, col4, col5 = st.columns(3)
area_code = col3.selectbox("Select Area Code", area_codes)
international_plan = col4.radio("International Plan", ["yes", "no"])
voice_mail_plan = col5.radio("Voice Mail Plan", ["yes", "no"])

col6, col7, col8 = st.columns(3)
number_vmail_messages = col6.number_input("Number of Voice Mail Messages", min_value=0, format="%d")
total_day_minutes = col7.number_input("Total Day Minutes", min_value=0.0, format="%f")
total_day_calls = col8.number_input("Total Day Calls", min_value=0, format="%d")

col9, col10, col11 = st.columns(3)
total_eve_minutes = col9.number_input("Total Evening Minutes", min_value=0.0, format="%f")
total_eve_calls = col10.number_input("Total Evening Calls", min_value=0, format="%d")
total_night_minutes = col11.number_input("Total Night Minutes", min_value=0.0, format="%f")

col12, col13, col14 = st.columns(3)
total_night_calls = col12.number_input("Total Night Calls", min_value=0, format="%d")
total_intl_minutes = col13.number_input("Total International Minutes", min_value=0.0, format="%f")
total_intl_calls = col14.number_input("Total International Calls", min_value=0, format="%d")

number_customer_service_calls = st.number_input("Number of Customer Service Calls", min_value=0, format="%d")

# Button to make predictions
if st.button("Predict Telecom customer churn"):
    
    input_data = {
        'state': [state],
        'account_length': [account_length],
        'area_code': [area_code],
        'international_plan': [international_plan],
        'voice_mail_plan': [voice_mail_plan],
        'number_vmail_messages': [number_vmail_messages],
        'total_day_minutes': [total_day_minutes],
        'total_day_calls': [total_day_calls],
        'total_eve_minutes': [total_eve_minutes],
        'total_eve_calls': [total_eve_calls],
        'total_night_minutes': [total_night_minutes],
        'total_night_calls': [total_night_calls],
        'total_intl_minutes': [total_intl_minutes],
        'total_intl_calls': [total_intl_calls],
        'number_customer_service_calls': [number_customer_service_calls]
    }
    

    # Create a DataFrame from the input data with consistent feature names
    input_data = pd.DataFrame(input_data)

    input_data['state'] = state_encoder.transform([input_data['state']])[0]
    input_data['area_code'] = area_code_encoder.transform([input_data['area_code']])[0]
    input_data['international_plan'] = international_plan_encoder.transform([input_data['international_plan']])[0]
    input_data['voice_mail_plan'] = voice_mail_plan_encoder.transform([input_data['voice_mail_plan']])[0]


    float_numerical_columns = [
    'account_length', 'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
    'total_eve_minutes', 'total_eve_calls', 'total_night_minutes', 'total_night_calls',
    'total_intl_minutes', 'total_intl_calls', 'number_customer_service_calls'
]


    # Apply standard scaling to the float numerical columns in X
    input_data[float_numerical_columns] = scaler.fit_transform(input_data[float_numerical_columns])
        
    
    # Load the pre-trained XGBoost model
        
    with open('Models/global_model.pkl', 'rb') as model_file:
        xgboost_model = pickle.load(model_file)

    # Make predictions
    churn_prediction = xgboost_model.predict(input_data)

    # Display prediction result
    if churn_prediction == 1:
        st.markdown("<h2 style='text-align: center; color : #E8570E;'>Churn Alert: Based on the provided information, it appears there might be a churn in this scenario.</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center; color : #008000;'>No Churn: Great news! According to our prediction, there is no indication of churn in this scenario. </h2>", unsafe_allow_html=True)
