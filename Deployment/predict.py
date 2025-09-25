import streamlit as st
import pandas as pd
import pickle

# Load trained model (keep the same filename as your previous script)

with open('model2_best.pkl', 'rb') as f:
    model = pickle.load(f)


def build_form():
    st.write('# Predict Customer Churn')
    st.write('### Isi dengan data pelanggan')

    with st.form('churn_form'):
        # Bagian profil & layanan
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox('Gender', ['Male', 'Female'])
            senior = st.selectbox('SeniorCitizen (0/1)', [0, 1], index=0)
            partner = st.selectbox('Partner', ['Yes', 'No'])
            dependents = st.selectbox('Dependents', ['Yes', 'No'])
            tenure = st.number_input('Tenure (bulan)', min_value=0, max_value=120, value=12, step=1)
            phone_service = st.selectbox('PhoneService', ['Yes', 'No'])
            multiple_lines = st.selectbox('MultipleLines', ['No', 'Yes', 'No phone service'])
        with col2:
            internet_service = st.selectbox('InternetService', ['Fiber optic', 'DSL', 'No'])
            online_security = st.selectbox('OnlineSecurity', ['Yes', 'No', 'No internet service'])
            online_backup = st.selectbox('OnlineBackup', ['Yes', 'No', 'No internet service'])
            device_protection = st.selectbox('DeviceProtection', ['Yes', 'No', 'No internet service'])
            tech_support = st.selectbox('TechSupport', ['Yes', 'No', 'No internet service'])
            streaming_tv = st.selectbox('StreamingTV', ['Yes', 'No', 'No internet service'])
            streaming_movies = st.selectbox('StreamingMovies', ['Yes', 'No', 'No internet service'])

        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox('PaperlessBilling', ['Yes', 'No'])
        payment_method = st.selectbox(
            'PaymentMethod',
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
        )
        monthly_charges = st.number_input('MonthlyCharges (US$)', min_value=0.0, value=70.35, step=0.05, format='%0.2f')
        total_charges = st.number_input('TotalCharges (US$)', min_value=0.0, value=850.50, step=0.50, format='%0.2f')

        submit = st.form_submit_button('Predict')

    # susun ke DataFrame sesuai contoh "data_baru"
    data_inf = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': int(senior),
        'Partner': partner,
        'Dependents': dependents,
        'tenure': int(tenure),
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': float(monthly_charges),
        'TotalCharges': float(total_charges),
    }])

    return submit, data_inf


def predict_and_show(df: pd.DataFrame):
    # Coba selaraskan urutan kolom dengan model jika punya atribut feature_names_in_
    try:
        if hasattr(model, 'feature_names_in_'):
            cols = list(model.feature_names_in_)
            missing = [c for c in cols if c not in df.columns]
            extra = [c for c in df.columns if c not in cols]
            if missing:
                st.warning(f'Kolom hilang pada input: {missing}')
            if extra:
                st.info(f'Kolom tambahan yang tidak dipakai model: {extra}')
            df = df.reindex(columns=cols, fill_value=0)
    except Exception as _:
        pass

    # Prediksi
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0]
            churn_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            st.metric('Probabilitas Churn', f'{churn_prob*100:.2f}%')
        pred = model.predict(df)[0]
        st.write(f'**Prediksi:** {pred}')
    except Exception as e:
        st.error('Terjadi error saat inferensi. Berikut detailnya:')
        st.exception(e)

    st.write('### Data Inference')
    st.dataframe(df.T.reset_index().rename(columns={'index': 'Field', 0: 'Value'}), height=520)


def run():
    submit, df = build_form()
    if submit:
        predict_and_show(df)


if __name__ == '__main__':
    run()
