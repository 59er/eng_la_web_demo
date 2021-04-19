import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import japanize_matplotlib
from pandas import plotting
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis as fa
from factor_analyzer import FactorAnalyzer 
import seaborn as sns 
import streamlit as st
import base64

def run_factor_app():
    st.header('■Factor analysis')

    st.sidebar.subheader('Data Upload')
    st.write('To create and analyze a class evaluation questionnaire.')

    df_edu = pd.read_csv("data/eng_sample_data_factor.csv")
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False, encoding = 'utf_8_sig')
            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    tmp_download_link = download_link(df_edu, 'sample_factor.csv', 'Download sample csv file.')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)

#     st.sidebar.info("""
#     [Download the sample csv file](https://github.com/59er/eng_learning_analytics_web/blob/master/sample_data/eng_sample_data_factor_for_WEB.csv)
#         """)

    uploaded_file = st.sidebar.file_uploader("File upload (Drag and drop or use [Browse files] button to import csv file. Only utf-8 format is available.)", type=["csv"])
	# uploaded_file = st.file_uploader(
	#     label = 'File Upload（Drag and drop csv/Excel）',
	#     type = ['csv', 'xlsx']
	# )

    try:

        if uploaded_file is not None:
            df_edu = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            display_data = st.sidebar.checkbox(label = 'Show uploaded data')
            
            if display_data:
                st.dataframe(df_edu)

        else:
            df_edu = pd.read_csv('data/eng_sample_data_factor.csv')

            show_df = st.sidebar.checkbox('Show DataFreme')

            if show_df == True:
                st.write(df_edu) 

        df = df_edu.copy()
        df = df.drop(['ID'], axis = 1)
        st.subheader('Data overview (correlation coefficient)')
        st.write(df.corr().style.background_gradient(cmap = 'coolwarm'))

        fa = FactorAnalyzer(n_factors = 3, rotation='promax', impute = 'drop')
        fa.fit(df)
        df_result = pd.DataFrame(fa.loadings_, columns = ['1st factor', '2nd factor', '3rd factor'], index = [df.columns])

        st.subheader('Factor Analysis Results')
        cm = sns.light_palette('blue', as_cmap=True)
        st.write(df_result.style.background_gradient(cmap = cm))

    except Exception as e:
        st.header('ERROR: Data inconsistency. Check data format to be uploaded.')
        print('Data inconsistency error')
