import streamlit as st
import pandas as pd
import joblib

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Customer Churn Prediction")
st.write("CSV upload karo — model batayega kaun churn karega")

uploaded_file = st.file_uploader("CSV file upload karo", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df.head())

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    df = df.drop(columns=['customerID'], errors='ignore')

    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)

    df = pd.get_dummies(df, columns=[
        'gender', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ])

    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    predictions = model.predict(df)
    df['Churn Prediction'] = predictions
    df['Churn Prediction'] = df['Churn Prediction'].map({1: 'Churn Karega ❌', 0: 'Nahi Karega ✅'})

    st.subheader("Predictions:")
    st.dataframe(df[['Churn Prediction']])

    churn_count = (predictions == 1).sum()
    safe_count = (predictions == 0).sum()

    col1, col2 = st.columns(2)
    col1.metric("Churn Karega", churn_count)
    col2.metric("Nahi Karega", safe_count)
