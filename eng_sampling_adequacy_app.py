import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import streamlit as st 
import seaborn as sns

def run_sampling_adequacy_app():

    st.header('■Measure of Sampling Adequacy')
    st.write('To investigate the adequay of the number of samples for questionnaire.Kaiser-Meyer-Olkin (KMO) Test is used.')
    st.sidebar.subheader('Data Upload')
    st.sidebar.info("""
    [Download the sample csv file](https://www.google.co.jp)
            """)
    try:

        uploaded_file = st.sidebar.file_uploader("File upload (Drag and drop or use [Browse files] button to import csv file. Only utf-8 format is available.)", type=["csv"])
        # uploaded_file = st.file_uploader(
        #     label = 'File Upload（Drag and drop csv/Excel file）',
        #     type = ['csv', 'xlsx']
        # )
        if uploaded_file is not None:
            df_edu = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            display_data = st.sidebar.checkbox(label = 'Show uploaded data')
            
            if display_data:
                st.dataframe(df_edu)

        else:
            df_edu = pd.read_csv('data/eng_sample_data_sampling.csv')

            show_df = st.sidebar.checkbox('Show DataFrame')

            if show_df == True:
                st.write(df_edu)

        df_edu = df_edu.dropna()
        df_edu = df_edu.drop(['student'],axis=1)
        from factor_analyzer.factor_analyzer import calculate_kmo
        kmo_all,kmo_model=calculate_kmo(df_edu)
        st.write('## KMO value:',kmo_model.round(2))

        st.subheader('Data overview (correlation coefficient)')
        st.write(df_edu.corr().style.background_gradient(cmap = 'coolwarm'))

        fa = FactorAnalyzer()
        fa.fit(df_edu)
        ev, v = fa.get_eigenvalues()

        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(7,5))
        plt.scatter(range(1,df_edu.shape[1]+1),ev)
        plt.plot(range(1,df_edu.shape[1]+1),ev)
        plt.title('Scree Plot')
        plt.xlabel('Factors')
        plt.ylabel('Eigenvalue')
        plt.grid()
        st.pyplot()

        fa = FactorAnalyzer(n_factors = 3, rotation='promax', impute='drop')
        fa.fit(df_edu)
        df_result = pd.DataFrame(fa.loadings_, columns = ['1st','2nd','3rd'])
        df_result.index = df_edu.columns
        cm = sns.light_palette('blue', as_cmap=True)
        df_factor = df_result.style.background_gradient(cmap = cm)
        st.write(df_factor)

    except Exception as e:
        st.header('ERROR: Data inconsistency. Check data format to be uploaded.')
        print('Data inconsistency error')