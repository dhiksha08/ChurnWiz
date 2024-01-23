import streamlit as st

def main():

    st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")
    st.markdown("<h1 style='text-align: center; color : #E8570E; font-size:65px'>Telecom Customer Churn Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("Project Features")
    st.markdown(
        """
        ## Handling Imbalanced Data with SMOTE

        To address the issue of class imbalance in the dataset, we applied the Synthetic Minority Over-sampling Technique (SMOTE). SMOTE is a resampling technique that creates synthetic samples in the minority class, thereby balancing the class distribution. This approach helps improve the performance and reliability of our predictive models, ensuring that they are not biased towards the majority class.

        ##Complete Model  
        In a separate analysis with the complete raw data, where no privacy-preserving techniques were applied, our model achieved exceptional accuracy.

        **Global Model Accuracy (Complete Data):** 99.04%

         ## Privacy-Preserving Techniques: Federated Learning and PATE

        Our project incorporates advanced privacy-preserving techniques to ensure the confidentiality and security of user data. We utilize Federated Learning and Private Aggregation of Teacher Ensembles (PATE) to safeguard sensitive information while still providing accurate predictions.

        ### Federated Learning
        In Federated Learning, models are trained collaboratively across multiple decentralized edge devices (clients) without exchanging raw data. Each client independently computes model updates on its local data and shares only these updates with the central server. This approach maintains data privacy as raw information never leaves the client's device.

        **Global Model Accuracy (Federated Learning):** 98.56%

        ### Private Aggregation of Teacher Ensembles (PATE)
        PATE introduces privacy into the model training process by adding noise to the aggregated predictions, ensuring individual data points remain confidential. It employs a teacher-student framework, where multiple teacher models make predictions on private data, and a student model learns from these noisy teacher predictions.

        - **Aggregated Predictions Accuracy (with noise):** 89.46%
        - **Aggregated Predictions Accuracy (without noise):** 95.82%

        ## Hyperparameter Tuning with Grid Search
        To enhance the accuracy of our models, we employed a rigorous hyperparameter tuning process using Grid Search. Grid Search systematically explores a range of hyperparameter values, allowing us to fine-tune our models and achieve optimal performance.

        **Grid Search Space:**
        - **n_estimators:** [50, 100, 200, 300]
        - **learning_rate:** [0.01, 0.05, 0.1]
        - **max_depth:** [3, 4, 5]
        - **subsample:** [0.8, 0.9, 1.0]
        - **colsample_bytree:** [0.8, 0.9, 1.0]
        - **alpha:** [0, 0.1, 1, 2]  *(L1 regularization term)*
        - **lambda:** [0, 0.1, 1, 2]  *(L2 regularization term)*

        **Best Hyperparameters Chosen:**
        | Hyperparameter    | Chosen Value |
        | ----------------- | ------------ |
        | n_estimators      | 300          |
        | learning_rate     | 0.1          |
        | max_depth         | 5            |
        | subsample         | 0.9          |
        | colsample_bytree  | 1.0          |
        | alpha             | 0.1          |
        | lambda            | 0.1          |

        ## Conclusion
        Our project combines cutting-edge privacy-preserving techniques with meticulous hyperparameter tuning, resulting in a global model accuracy of 99.04%. This approach ensures the privacy of user data while delivering highly accurate predictions.
        """
    )

if __name__ == "__main__":
    main()
