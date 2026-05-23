import streamlit as st
import pandas as pd
import joblib

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Churn Predictor", page_icon="📊")

st.title("📊 Customer Churn Predictor")
st.markdown("Upload your customer data to identify who is likely to cancel their subscription.")
st.divider()

uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"Total customers loaded: {len(df)}")
    st.divider()

    df_original = df.copy()
    df = df.drop(columns=['customerID', 'Churn'], errors='ignore')

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

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

    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns]

    predictions = model.predict(df)
    df_original['Churn Prediction'] = predictions
    df_original['Churn Prediction'] = df_original['Churn Prediction'].map({
        1: '⚠️ At Risk',
        0: '✅ Likely to Stay'
    })

    churn_count = int((predictions == 1).sum())
    safe_count = int((predictions == 0).sum())
    churn_rate = round((churn_count / len(predictions)) * 100, 1)

    st.subheader("📈 Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(predictions))
    col2.metric("At Risk", churn_count, delta=f"{churn_rate}%", delta_color="inverse")
    col3.metric("Likely to Stay", safe_count)

    st.divider()

    st.subheader("🔍 Customer-wise Predictions")
    at_risk = df_original[df_original['Churn Prediction'] == '⚠️ At Risk']
    staying = df_original[df_original['Churn Prediction'] == '✅ Likely to Stay']

    tab1, tab2 = st.tabs([f"⚠️ At Risk ({churn_count})", f"✅ Likely to Stay ({safe_count})"])

    with tab1:
        st.dataframe(at_risk[['customerID', 'Churn Prediction']], use_container_width=True)

    with tab2:
        st.dataframe(staying[['customerID', 'Churn Prediction']], use_container_width=True)

    st.divider()
    st.download_button(
        label="⬇️ Download Full Report",
        data=df_original[['customerID', 'Churn Prediction']].to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv"
    )
