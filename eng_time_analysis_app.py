from warnings import filterwarnings
import statsmodels.api as sm
import pandas as pd
import numpy as np
import requests
import io
from matplotlib import pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import base64

def run_time_analysis_app():

    st.header('■Time series analysis')
    st.write('To visualize the academic achievement over time.')

    st.sidebar.subheader('Data Upload')
    
    df_edu = pd.read_csv("data/eng_sample_data_time_analysis.csv")
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False, encoding = 'utf_8_sig')
            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    tmp_download_link = download_link(df_edu, 'sample_time_analysis.csv', 'Download sample csv file.')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)

#     st.sidebar.info("""
#     [Download the sample csv file](https://github.com/59er/eng_learning_analytics_web/blob/master/sample_data/eng_sample_data_time_analysis_for_WEB.csv)""")
    
    
    st.sidebar.subheader('Data upload')

    try:

        uploaded_file = st.sidebar.file_uploader(
            label='File upload (Drag and drop or use [Browse files] button to import csv file. Only utf-8 format is available.)',
            type=['csv']
            #type=['csv','xlsx']
        )

        if uploaded_file is not None:
            df=pd.read_csv(uploaded_file)
            display_data = st.sidebar.checkbox(label='Show uploaded data')

            if display_data:
                st.dataframe(df)
        else:

            data = pd.read_csv('data/eng_sample_data_time_analysis.csv', index_col='Date', parse_dates=True)
            display_data = st.sidebar.checkbox(label='Show DataFrame')

            if display_data:
                st.dataframe(data)

        #data = pd.read_csv('data/eng_sample_data_time_analysis.csv', index_col='Date', parse_dates=True)

        data['Total learning time (min.)'] = data['Total learning time (min.)'].astype('float64')

        st.subheader('Scoring history')

        class_list = data['Class'].unique()
        option_class = st.selectbox(
            'Class：Select Class',
            (class_list)

        )

        df_mean = data.groupby(['Class']).mean()
        df_mean = data[data['Class'] == option_class]

        st.write('Progress in overall average score')

        fig = plt.figure(figsize=(15,6))
        plt.plot(df_mean['Mean Score'])
        plt.tight_layout()
        plt.show()
        st.pyplot()

        st.subheader('Relationship between learning time and average score')
        import matplotlib.dates as dt
        import japanize_matplotlib
        x = df_mean.index
        volume = df_mean['Total learning time (min.)']
        price = df_mean['Mean Score']
        fig, ax1 = plt.subplots(figsize = (15,6))
        ax1.bar(x, volume, color = 'darkcyan', label='Learning time',width = 10)
        ax1.set_ylabel('Learning time',fontsize=14)
        ax2 = ax1.twinx()
        ax2.plot(x, price,color='red', label='Mean Score')
        ax2.set_ylabel('Learning time', fontsize=14)
        # ax2.xaxis.set_major_formatter(dt.DateFormatter('%Y/%m'))
        # ax2.xaxis.set_major_locator(dt.DayLocator(interval=7))
        plt.title('Relationship between learning time and average score', fontsize=16)
        ax1.legend(bbox_to_anchor=(1, 1), loc='lower left', frameon=False, fontsize=14)
        ax2.legend(bbox_to_anchor=(1, 0.9), loc = 'lower left', borderaxespad=0, frameon=False, fontsize=14)
        plt.tight_layout()
        plt.show()
        st.pyplot()

        st.subheader('Relationship between learning time and average score when learning time is shifted by one month.')
        fig, ax1 = plt.subplots(figsize=(15,6))
        ax1.bar(x,volume.shift(1),color='coral', label='Total learning time (min.)', width=10)
        ax1.set_ylim(0,2000)
        ax1.set_ylabel('Total learning time (min.)', fontsize=14)
        ax2 = ax1.twinx()
        ax2.plot(price, label='Mean Score')
        ax2.set_ylim(45,65)
        ax2.set_ylabel('Mean Score', fontsize=14)
        #ax2.xaxis.set_major_formatter(dt.DateFormatter('%Y:%M'))
        ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', frameon=False, fontsize=14)
        ax2.legend(bbox_to_anchor=(1, 0.9),  loc='upper right',borderaxespad=0, frameon=False, fontsize=14)
        # ax1.yaxis.set_minor_locator(MultipleLocator(0.5))
        # ax2.yaxis.set_minor_locator(MultipleLocator(5))
        plt.title('Relationship between learning time and average score when learning time is shifted by one month.', fontsize=16)
        plt.grid()
        plt.tight_layout()
        plt.show()
        st.pyplot()


    except Exception as e:
        st.header('ERROR: Data inconsistency. Check data format to be uploaded.')
        print('Data inconsistency error')
