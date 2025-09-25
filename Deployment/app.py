# import libraries
import streamlit as st
import eda, predict

# bagian dalam sidebar
with st.sidebar:
    st.write("# Page Navigation")

    # toggle pilih halaman
    page = st.selectbox("Pilih Halaman", ("EDA", 'Predict Churn'))

    # test
    st.write(f'Halaman yang di tampilkan: {page}')

    st.write('## About')
    # magic
    '''
    Page ini berisikan hasil analisis churn pelanggan Telco.
    Ini dibuat untuk membantu memprediksi apakah pelanggan akan churn atau tidak.

    '''

# bagian luar sidebar
if page == 'EDA':
    eda.run()

else:
    predict.run()