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

    # Original df se customerID aur Churn hata do
    df_original = df.copy()
    df = df.drop(columns=['customerID', 'Churn'], errors='ignore')

    # TotalCharges fix
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Binary columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)

    # Get dummies
    df = pd.get_dummies(df, columns=[
        'gender', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ])

    # Scaling
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Missing columns add karo jo training mein thi
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Sirf model wale columns rakho — same order mein
    df = df[model_columns]

    # Prediction
    predictions = model.predict(df)
    df_original['Churn Prediction'] = predictions
    df_original['Churn Prediction'] = df_original['Churn Prediction'].map({
        1: 'Churn Karega ❌',
        0: 'Nahi Karega ✅'
    })

    # Result
    st.subheader("Predictions:")
    st.dataframe(df_original[['customerID', 'Churn Prediction']])

    churn_count = (predictions == 1).sum()
    safe_count = (predictions == 0).sum()

    col1, col2 = st.columns(2)
    col1.metric("Churn Karega ❌", churn_count)
    col2.metric("Nahi Karega ✅", safe_count)
