import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns 
import japanize_matplotlib
import base64

def run_cluster_app():

    st.header('■Cluster Analysis')
    st.write('- To group students with similar learning characteristics.')
    
    st.sidebar.subheader('Data upload')
    df_edu = pd.read_csv("data/eng_sample_data_cluster.csv")
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False, encoding = 'utf_8_sig')
            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    tmp_download_link = download_link(df_edu, 'sample_cluster.csv', 'Download sample csv file.')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)
    
#     st.sidebar.info("""
#     [Download the sample csv file](https://github.com/59er/eng_learning_analytics_web/blob/master/sample_data/eng_sample_data_cluster_for_WEB.csv)
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
            display_data = st.sidebar.checkbox(label = 'Show uploaded data')
            
            if display_data:
                st.dataframe(df_edu)

            # df_edu.columns = ['','Literature','History','Math','Physics']
            # st.write(df_edu.columns)
            dat_i = df_edu.set_index('ID')
            dat_i.describe().round(2)
            dat_i.mean(axis=1)
            pred = KMeans(n_clusters = 3).fit_predict(dat_i)
            dat_i1 = dat_i.copy()
            dat_i1['cluster_id'] = pred
            dat_i1['cluster_id'].value_counts()

            # z = linkage(dat_i, metric = 'euclidean', method = 'ward')
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # fig = plt.figure(figsize = (12,6), facecolor = 'w')
            # ax = fig.add_subplot(title= '樹形図: 全体')
            # dendrogram(z)
            # st.pyplot(fig)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(sns.clustermap(dat_i,col_cluster = False, cmap='Blues', linewidth=.5))
            st.pyplot()    


        else:
            df_edu = pd.read_csv('data/eng_sample_data_cluster.csv')
            show_df = st.sidebar.checkbox('Show DataFreme')
            if show_df == True:
                st.write(df_edu)

            df_edu.columns = ['','Literature','Reading','History','Math','Physics']
            dat_i = df_edu.set_index('')
            dat_i.describe().round(2)
            dat_i.mean(axis=1)
            pred = KMeans(n_clusters = 3).fit_predict(dat_i)
            dat_i1 = dat_i.copy()
            dat_i1['cluster_id'] = pred
            dat_i1['cluster_id'].value_counts()

            # z = linkage(dat_i, metric = 'euclidean', method = 'ward')
            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # fig = plt.figure(figsize = (12,6), facecolor = 'w')
            # ax = fig.add_subplot(title= '樹形図: 全体')
            # dendrogram(z)
            # st.pyplot(fig)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(sns.clustermap(dat_i,col_cluster = False, cmap='Blues', linewidth=.5))
            st.pyplot()

            st.write('This example suggests that there are two groups: those who are good at humanities subjects and those who are good at science subjects.')
        
    except Exception as e:
        st.header('ERROR: Data inconsistency. Check data format to be uploaded.')
        print('Data inconsistency error')
