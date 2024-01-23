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


# Set Streamlit theme to dark, enable "Run on Save," and set default layout to wide
st.set_page_config(page_title="Telecom Churn Prediction", page_icon="🚕", layout="wide")
st.markdown("<h1 style='text-align: center; color : #E8570E; font-size:65px'>Telecom Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<br><hr><br>", unsafe_allow_html=True)

train=pd.read_csv("Dataset/train.csv")
test=pd.read_csv("Dataset/test.csv")



st.header("Understanding the Data")

string_columns = train.select_dtypes(include=['object'])
plt.style.use('dark_background')
st.subheader('State-wise count of churned Customers') # Add topic name
for column in string_columns:
    if column == "state":
        fig, ax = plt.subplots(figsize=(20, 5))
        sns.countplot(x=column, data=string_columns, ax=ax)
        st.pyplot(fig)
st.write (
    """ **Inference** - West Virginia has the highest count
                        California has the lowest count"""
)

plt.style.use('dark_background')
st.subheader('Customer churn in training data') # Add topic name

trace = go.Pie(labels=train["churn"].value_counts().keys().tolist(),
               values=train["churn"].value_counts().values.tolist(),
               marker=dict(colors=['royalblue', 'lime'],
                           line=dict(color="white", width=1.3)),
               rotation=90,
               hoverinfo="label+value+text",
               hole=0.5
               )

# Create the layout
layout = go.Layout(dict(
                        plot_bgcolor="rgb(243, 243, 243)",
                        paper_bgcolor="rgb(243, 243, 243)"))

# Create the figure
fig = go.Figure(data=[trace], layout=layout)

# Display the chart using Streamlit
st.plotly_chart(fig)
st.write (
    """ **Inference** - The dataset is highly imbalaced.\n
    'No  : '  85.9 % of the dataset\n 
    'Yes : ' 14.1% of the dataset """
)


plt.style.use('dark_background')
st.subheader('Total Day charge v/s Total DayCalls') # Add topic name
fig, ax = plt.subplots(figsize=(15, 5))
sns.scatterplot(data=train, x='total_day_calls', y='total_day_charge', palette='viridis')
plt.xlabel('Total Day Calls', fontsize=10, fontweight='bold')
plt.ylabel('Total Day Charge', fontsize=10, fontweight='bold')
plt.title('Day charge v/s Calls', fontsize=12, fontweight='bold')
st.pyplot(fig)
st.write (
    """ **Inference** - As the number of day calls increases, the total day charge also tends to increase.."""
)


plt.style.use('dark_background')
st.subheader('"Churn Analysis based on International Plan and Voice Mail Subscription') # Add topic name
fig, ax = plt.subplots(1,2,figsize=(16,5))
cat_cols = ['state','area_code','international_plan','voice_mail_plan','churn']
for col,subplot in zip(cat_cols[2:-1], ax.flatten()):
    f = (
        so.Plot(train, x=col, color='churn')
        .add(so.Bar(), so.Count(), so.Stack())
        .label(x=" ", y="Count of Customers", title=col)
        .on(subplot).plot()
    )
st.pyplot(fig)
#st.write (
#    """ **Inference** - West Virginia has the highest"""
#)


st.header("Outlier Analysis")

plt.style.use('dark_background')
st.subheader('Customer churn data') # Add topic name
num_cols = train.select_dtypes(include=['int', 'float']).columns

# Create subplots
fig, axes = plt.subplots(5, 3, figsize=(15, 10))
fig.suptitle("Outlier Check in Numerical Columns", fontsize=16)

for ax, col in zip(axes.flatten(), num_cols):
    ax.boxplot(train[col], vert=False)
    ax.set_title(f"{col} Box Plot")
    ax.set_ylabel(col)
    ax.set_xlabel("Values")

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Display the plot using Streamlit
st.pyplot(fig)

st.write("""
    Replaced the outliers with median value.
""")

plt.style.use('dark_background')
st.subheader('VISUALISING AFTER REPLACING OUTLIERS') # Add topic name
num_cols = train.select_dtypes(include=['int', 'float']).columns

fig, ax = plt.subplots(15, 2, figsize=(15, 60))
i = 0
color_dict = {'no': matplotlib.colors.to_rgba('cornflowerblue', 0.3),
              'yes': matplotlib.colors.to_rgba('crimson', 1)}
for col in num_cols:
    sns.histplot(data=train, x=col, hue='churn', ax=ax[i, 0], legend=True,
                 palette=color_dict, kde=True, fill=True)
    sns.boxplot(data=train, y=col, x='churn', ax=ax[i, 1],
                palette=('cornflowerblue', 'crimson'))
    ax[i, 0].set_ylabel(col, fontsize=12)
    ax[i, 0].set_xlabel(' ')
    ax[i, 1].set_xlabel(' ')
    ax[i, 1].set_ylabel(' ')
    ax[i, 1].xaxis.set_tick_params(labelsize=14)
    i += 1
plt.style.use('dark_background')
plt.tight_layout()

# Display the subplots using Streamlit
st.pyplot(fig)
#st.write (
#    """ **Inference** - West Virginia has the highest"""
#)


plt.style.use('dark_background')
st.subheader('Correlation Heatmap')
correlation_matrix = train.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.set(font_scale=0.7)
sns.set_style("whitegrid")
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", linewidths=1, square=True)
plt.title('Correlation Heatmap')
st.pyplot(fig)
#st.write (
#    """ **Inference** - West Virginia has the highest"""
#)


for data in [train, test]:
    data.drop(columns=['total_day_charge',
                       'total_eve_charge',
                      'total_night_charge',
                      'total_intl_charge'], inplace=True)

plt.style.use('dark_background')
st.subheader('Customer Service Calls and Churn Status')
plt.figure(figsize=(10, 6))
sns.countplot(x="number_customer_service_calls", hue="churn", data=train)
plt.title("Customer Churn by Number of Customer Service Calls")
plt.xlabel("Number of Customer Service Calls")
plt.ylabel("Count")
plt.legend(title="Churn", loc="upper right")
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
st.pyplot(plt)
st.write (
    """ **Inference** - The interaction and experience during the first call might significantly influence the customer's decision regarding their subscription."""
)