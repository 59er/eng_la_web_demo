import plotly.express as px 
import numpy as np 
import pandas as pd
import pydeck as pdk 
import streamlit as st
from ptitprince import RainCloud
import matplotlib.pyplot as plt
import base64

def run_edu_overview_app():

    df_edu = pd.read_csv('data/eng_sample_data_overview.csv')

    st.header('■Overview')
    st.write('To get an overview of the test results and to understand the trend of each class visually.')
        
    st.sidebar.subheader('Data Upload')
    
    df_edu = pd.read_csv("data/eng_sample_data_overview.csv")
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False, encoding = 'utf_8_sig')
            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    tmp_download_link = download_link(df_edu, 'sample_overview.csv', 'Download sample csv file.')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)

#     st.sidebar.info("""
#     [Download the sample csv file](https://github.com/59er/eng_learning_analytics_web/blob/master/sample_data/eng_sample_data_overview_for_web_without_G.csv)
#         """)
    try:
        uploaded_file = st.sidebar.file_uploader("File upload (Drag and drop or use [Browse files] button to import csv file. Only utf-8 format is available.）", type=["csv"])
        # uploaded_file = st.file_uploader(
        #     label = 'File Upload（drag and drop csv/Excel）',
        #     type = ['csv', 'xlsx']
        # )
        if uploaded_file is not None:
            df_edu = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            display_data = st.sidebar.checkbox(label = 'Show uploaded data')
            
            if display_data:
                st.dataframe(df_edu)

        else:
            df_edu = pd.read_csv('data/eng_sample_data_overview.csv')

            show_df = st.sidebar.checkbox('Show DataFreme')
            if show_df == True:
                st.write(df_edu)


        st.subheader('Score distribution for each subject')

    #Select subject
        subject_list = df_edu['Subject'].unique()
        option_subject = st.selectbox(
            'Subject：Select a subject.',
            (subject_list)
        )
        df_subject = df_edu[df_edu['Subject']== option_subject]
        st.set_option('deprecation.showPyplotGlobalUse', False)
    #Raincloud by subject
        fig, ax = plt.subplots(figsize=(10, 3))
        RainCloud(data=df_subject, y='Score', orient='h', ax=ax, width_viol=1)
        plt.tight_layout()
        ax.grid()
        st.pyplot()
    #Mean
        df_subject_mean = df_edu.groupby('Subject').mean()
        df_subject_mean.rename(columns = {'Score':'Mean'}, inplace = True)
        df_subject_mean = df_subject_mean.reset_index()
        df_subject_mean = df_subject_mean[df_subject_mean['Subject'] == option_subject]    
        st.dataframe(df_subject_mean.round(2))

        st.subheader('Score distribution for each class')
        st.text('* The average points are connected by a dotted line')
    #Score by class

        fig, ax = plt.subplots(figsize=(20, 10))
        plt.title("Score by class")
        plt.xlabel('Class')
        plt.ylabel('Score')
        #To display mean
        ax = RainCloud(data=df_subject, y='Score', x='Class', ax=ax, pointplot = True, 
        point_linestyles='--',
        linecolor='tab:blue',
        width_viol=0.7,width_box=0.2,box_linewidth=1,point_size=6,rain_alpha=1)

        plt.tight_layout()
        ax.grid()
        st.pyplot()

        st.subheader('Test details for each class')
        class_list = df_edu['Class'].unique()
        option_class = st.selectbox(
            'Class：Select a class.',
            (class_list)
        )
        #Mean
        df_mean = df_edu.groupby(['Class','Subject','Teacher']).mean()
        df_mean.rename(columns = {'Score':'Mean'}, inplace = True)
        df_mean = df_mean.reset_index()
        df_mean = df_mean[df_mean['Class'] == option_class]
        #Variance
        df_var = df_edu.groupby(['Class','Subject','Teacher']).var()
        df_var.rename(columns = {'Score':'Variance'}, inplace = True)
        df_var = df_var.reset_index()
        df_var = df_var[df_var['Class'] == option_class]
    
        st.subheader('・Mean and variance')

        st.dataframe(df_mean.round(2))
        st.dataframe(df_var.round(2))

        st.subheader('・Score Distribution')
        #Histgram for selected class

        df_subject_class = df_subject[df_subject['Class'] == option_class]
        plt.figure(figsize = (5,7))
        fig = df_subject_class.hist(alpha = 0.5)
        plt.tight_layout()
        st.pyplot()

    except Exception as e:
        st.header('ERROR: Data inconsistency. Check data format to be uploaded.')
        print('Data inconsistency error')
