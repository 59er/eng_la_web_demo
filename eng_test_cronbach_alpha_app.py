import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import streamlit as st
import base64

def run_test_cronbach_alpha_app():

    st.header('■Exam questions/questionnaire reliability measurement (Cronbach α)')
    st.write('To measure the validity and reliability of tests.')
    	
    st.sidebar.subheader('Data Upload')
    
    df_edu = pd.read_csv("data/eng_sample_data_cronbach.csv")
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False, encoding = 'utf_8_sig')
            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    tmp_download_link = download_link(df_edu, 'sample_cronbach.csv', 'Download sample csv file.')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)

#     st.sidebar.info("""
#     [Download the sample csv file](https://github.com/59er/eng_learning_analytics_web/blob/master/sample_data/eng_sample_data_cronbach_for_WEB.csv)
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

            data = df_edu.drop(['ID'], axis=1)
            df_y = data.var(axis=0)
            df_y_sum = df_y.sum()
            df_c_sum= data.sum(axis=1)
            df_c_sum_var = df_c_sum.var()
            alpha = (len(data.columns)/(len(data.columns)-1))*(1-(df_y_sum/df_c_sum_var))
            st.write('・Reliability coefficient(Cronbach α)：',alpha.round(3))


            A = []

            n_item = len(data.columns)

            for i in data.columns:
                data_x = data.drop(i, axis=1)
                item_var = sum(data_x.var())
                total_var = data_x.sum(axis=1).var()
                a = n_item / (n_item - 1) * (1-(item_var / total_var))
                A.append(a)
            A = pd.DataFrame(A)
            # index =['Item 1','Item 2','Item 3','Item 4','Item 5']
            index = data.columns
            A['index'] = index
            A = A.loc[:, ['index', 0]]
            A = A.rename(columns = {'index':'Item number',0:'Reliability coefficient'})
            A = A.sort_values(by = 'Reliability coefficient',ascending = False)
            A = A.round(3)
            st.write('・Reliability result if dropped each item')
            st.dataframe(A.style.highlight_max(axis=0),width=500,height=500)
            # st.write(A)

            st.write('* The result of finding the reliability coefficient by dropping the quiz of the item number is shown.\
            If the quiz is deleted and the reliability coefficient is high, it is considered to be a factor that lowers the overall reliability.')


        else:
            data = pd.read_csv('data/eng_sample_data_cronbach.csv')
            data = data.drop(['ID'], axis=1)

            show_df = st.sidebar.checkbox('Show DataFreme')

            if show_df == True:
                st.write(data)

            df_y = data.var(axis=0)
            df_y_sum = df_y.sum()
            df_c_sum= data.sum(axis=1)
            df_c_sum_var = df_c_sum.var()
            alpha = (len(data.columns)/(len(data.columns)-1))*(1-(df_y_sum/df_c_sum_var))
            st.write('・Reliability coefficient(Cronbach α)：',alpha.round(3))


            A = []

            n_item = len(data.columns)

            for i in data.columns:
                data_x = data.drop(i, axis=1)
                item_var = sum(data_x.var())
                total_var = data_x.sum(axis=1).var()
                a = n_item / (n_item - 1) * (1-(item_var / total_var))
                A.append(a)
            A = pd.DataFrame(A)
            index =['Item 1','Item 2','Item 3','Item 4','Item 5']
            A['index'] = index
            A = A.loc[:, ['index', 0]]
            A = A.rename(columns = {'index':'Item number',0:'Reliability coefficient'})
            A = A.sort_values(by = 'Reliability coefficient',ascending = False)
            A = A.round(3)
            st.write('・Reliability result if dropped each item')
            st.dataframe(A.style.highlight_max(axis=0),width=500,height=500)
            # st.write(A)

            st.write('* The result of finding the reliability coefficient by dropping the quiz of the item number is shown.\
            If the quiz is deleted and the reliability coefficient is high, it is considered to be a factor that lowers the overall reliability.')

    except Exception as e:
        st.header('ERROR: Data inconsistency. Check data format to be uploaded.')
        print('Data inconsistency error')
