import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import japanize_matplotlib 
import scipy as sp
from sklearn import linear_model
import base64

def run_corr_app():

    st.header('■Correlation analysis')
    st.write('To investigate the relationship between midterm and final exam grades and so on.')
    st.sidebar.subheader('Data Upload')
    
    df_edu = pd.read_csv("data/eng_sample_data_corr.csv")
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False, encoding = 'utf_8_sig')
            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    tmp_download_link = download_link(df_edu, 'sample_corr.csv', 'Download sample csv file.')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)

#     st.sidebar.info("""
#     [Download the sample csv file](https://github.com/59er/eng_learning_analytics_web/blob/master/sample_data/eng_sample_data_corr_for_WEB.csv)
#         """)
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
            df_edu = pd.read_csv('data/eng_sample_data_corr.csv')

            show_df = st.sidebar.checkbox('Show DataFrame')

            if show_df == True:
                st.write(df_edu)
        
        numeric_columns = list(df_edu.select_dtypes(['int32','int64','float32','float64']).columns)
        non_numeric_columns = list(df_edu.select_dtypes(['object']).columns)

        st.subheader('Select X axis and Y axis')
        x_value = st.selectbox(label = 'X axis', options = numeric_columns)
        y_value = st.selectbox(label = 'Y axis', options = numeric_columns)

        X = df_edu[x_value].values
        Y = df_edu.loc[:,[y_value]].values
        REG = linear_model.LinearRegression()
        REG.fit(Y,X)

        st.write('Regression coefficient:', REG.coef_)
        st.write('Intercept :', REG.intercept_)

        plt.figure(figsize = (8,5))

        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.scatter(X,Y)
        plt.plot(Y, REG.predict(Y))
        plt.grid(True)
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        st.pyplot()

        st.write('Coefficient of determination:',REG.score(Y,X))
        st.write('Correlation coefficient/P-Value:',sp.stats.pearsonr(df_edu[x_value],df_edu[y_value]))

    except Exception as e:
        st.header('ERROR: Data inconsistency. Check data format to be uploaded.')
        print('Data inconsistency error')
