import pandas as pd
from scipy import stats 
import matplotlib.pyplot as plt 
import streamlit as st
import base64

def run_test_difficulty_app():

    st.header('■Exam questions difficulty and discrimination analysis')
    st.write('To grade each test based on its difficulty and discrimination.')
    	
    st.sidebar.subheader('Data Upload')
    
    df_edu = pd.read_csv("data/ja_sample_data_difficulty_Q20.csv")
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False, encoding = 'utf_8_sig')
            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    tmp_download_link = download_link(df_edu, 'sample_diff.csv', 'Download sample csv file.')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)

#     st.sidebar.info("""
#     [Download the sample csv file](https://github.com/59er/eng_learning_analytics_web/blob/master/sample_data/eng_sample_data_difficulty_for_WEB.csv)
#         """)
    try:
        uploaded_file = st.sidebar.file_uploader("File upload (Drag and drop or use [Browse files] button to import csv file. Only utf-8 format is available.)", type=["csv"])
        # uploaded_file = st.file_uploader(
        #     label = 'File Upload（csv/Excelファイルをドラック）',
        #     type = ['csv', 'xlsx']
        # )
        if uploaded_file is not None:
            df_edu = pd.read_csv(uploaded_file)
            display_data = st.sidebar.checkbox(label = 'Show uploaded data')
            
            if display_data:
                st.dataframe(df_edu)

            df_edu = df_edu.drop(['ID'], axis=1)
            #項目難易度
            item_diff = df_edu.mean()

            #識別力
            R_pbi = []
            for i in df_edu.columns:
                r = stats.pointbiserialr(df_edu[i], df_edu.sum(axis=1))
                R_pbi.append(r)

            D = []
            for i in range(len(R_pbi)):
                D.append(R_pbi[i][0])

            #ある項目を削除した場合の信頼性係数

            A = []
            # n_item = 50
            n_item = len(df_edu.columns)
            for i in df_edu.columns:
                data_x = df_edu.drop(i, axis=1)
                item_var = sum(data_x.var())
                total_var = data_x.sum(axis=1).var()
                a = n_item / (n_item-1) * (item_var / total_var)
                A.append(a)

            item_property = pd.DataFrame({'Difficulty': item_diff, 'Discrimination':D, 'Alpha_if_dropped':A}, index = df_edu.columns)
            st.dataframe(item_property)
            plt.figure(figsize = (5,5))
            for i,j,k,l in zip(item_diff, D, A, df_edu.columns):
                plt.scatter(i,j, c = 50, cmap='Blues', vmin=0.2, vmax = 0.99)
                plt.annotate(l,xy = (i,j))
            plt.xlabel('認識力')
            plt.ylabel('難易度')
            plt.colorbar()
            #item_property.plot(kind = 'scatter', x = 'Discrimination', y = 'Difficulty', c = 'Alpha_if_dropped')
            plt.show()
            st.pyplot()

            # for i, j, k in zip(item_diff, R_pbi, df_edu.columns):
            #     plt.plot(i,j,'o', color='k')
            #     plt.annotate(k,xy=(i,j))
            # plt.show()
            # st.pyplot()

            st.write('As for the color of the dots, the darker the color,\
                 the more the reliability coefficient (alpha) of the question will increase by removing that question.')

        else:
            df_edu = pd.read_csv('data/ja_sample_data_difficulty_Q20.csv',index_col=0)
        
            show_df = st.sidebar.checkbox('Show DataFrame')
            if show_df == True:
                st.write(df_edu)


            #項目難易度
            item_diff = df_edu.mean()

            #識別力
            R_pbi = []
            for i in df_edu.columns:
                r = stats.pointbiserialr(df_edu[i], df_edu.sum(axis=1))
                R_pbi.append(r)

            D = []
            for i in range(len(R_pbi)):
                D.append(R_pbi[i][0])

            #ある項目を削除した場合の信頼性係数

            A = []
            # n_item = 50
            n_item = len(df_edu.columns)
            for i in df_edu.columns:
                data_x = df_edu.drop(i, axis=1)
                item_var = sum(data_x.var())
                total_var = data_x.sum(axis=1).var()
                a = n_item / (n_item-1) * (item_var / total_var)
                A.append(a)

            item_property = pd.DataFrame({'Difficulty': item_diff, 'Discrimination':D, 'Alpha_if_dropped':A}, index = df_edu.columns)
            st.dataframe(item_property)
            plt.figure(figsize = (5,5))
            for i,j,k,l in zip(item_diff, D, A, df_edu.columns):
                plt.scatter(i,j, c = 50, cmap='Blues', vmin=0.2, vmax = 0.99)
                plt.annotate(l,xy = (i,j))
            plt.xlabel('Discrimination')
            plt.ylabel('Difficulty')
            plt.colorbar()
            #item_property.plot(kind = 'scatter', x = 'Discrimination', y = 'Difficulty', c = 'Alpha_if_dropped')
            plt.show()
            st.pyplot()

            # for i, j, k in zip(item_diff, R_pbi, df_edu.columns):
            #     plt.plot(i,j,'o', color='k')
            #     plt.annotate(k,xy=(i,j))
            # plt.show()
            # st.pyplot()


            st.write('As for the color of the dots, the darker the color,\
                 the more the reliability coefficient (alpha) of the question will increase by removing that question.')

    except Exception as e:
        st.header('ERROR: Data inconsistency. Check data format to be uploaded.')
        print('Data inconsistency error')

