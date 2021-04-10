import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.figure_factory as ff 
import japanize_matplotlib 
from sklearn import linear_model
import scipy as sp
import base64

def run_edu_detail_app():

    st.header('■Detailed analysis')
    st.write('To investigate the relationship between learning time and academic achievement.')

    st.sidebar.subheader('Input data')
    
    df_edu = pd.read_csv("data/eng_sample_data_detail.csv")
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False, encoding = 'utf_8_sig')
            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    tmp_download_link = download_link(df_edu, 'sample_detail.csv', 'Download sample csv file.')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)
    
#     st.sidebar.info("""
#     [Download the sample csv file](https://github.com/59er/eng_learning_analytics_web/blob/master/sample_data/eng_sample_data_detail_for_WEB.csv)
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
            display_data = st.sidebar.checkbox(label = 'Display uploaded file?')
            
            if display_data:
                st.dataframe(df_edu)

        else:
            df_edu = pd.read_csv('data/eng_sample_data_detail.csv')

            show_df = st.sidebar.checkbox('Show DataFreme')

            if show_df == True:
                st.write(df_edu)
        
        numeric_columns = list(df_edu.select_dtypes(['int32','int64','float32','float64']).columns)
        non_numeric_columns = list(df_edu.select_dtypes(['object']).columns)
        non_numeric_columns.pop(3)#drop Subject
        non_numeric_columns.pop(0)#drop ID
        # non_numeric_columns = non_numeric_columns.remove('Subject')
        submenu = st.selectbox("Submenu",["Learning time(by class/by subject/by teacher)",
                                            "Correlation between learning time and score (by subject)"])


    #Select subject
        subject_list = df_edu['Subject'].unique()
        option_subject = st.selectbox(
            'Subject：Select subject',
            (subject_list)
        )
        df_subject = df_edu[df_edu['Subject']== option_subject]


        if submenu == "Learning time(by class/by subject/by teacher)":
            st.subheader("■Learning time(By class/by subject/by teacher)")
            st.subheader('Select X axis and Y axis')
            x_value = st.selectbox(label = 'X axis', options = non_numeric_columns)
            y_value = st.selectbox(label = 'Y axis', options = numeric_columns)
            fig = px.bar(df_subject, x = x_value, y = y_value)

            st.plotly_chart(fig)

            #Mean
            df_mean = df_subject.groupby(['Class','Teacher']).mean()
            df_mean.rename(columns = {'Score':'Mean'}, inplace = True)
            df_mean = df_mean.reset_index()
            # df_mean = df_mean[df_mean['Class'] == option_class]
            #Variance
            df_var = df_subject.groupby(['Class','Teacher']).var()
            df_var.rename(columns = {'Score':'Variance'}, inplace = True)
            df_var = df_var.reset_index()
            # df_var = df_var[df_var['Class'] == option_class]
        
            st.subheader('・Learning time per person and average score for each class')

            st.dataframe(df_mean.round(2))

        elif submenu == "Correlation between learning time and score (by subject)":    
            st.subheader('・Correlation between learning time and score (by subject)')

            # st.subheader('Select X and Y')
            # x_value = st.selectbox(label = 'X axis', options = non_numeric_columns)
            # y_value = st.selectbox(label = 'Y axis', options = numeric_columns)

            X = df_subject['Learning time'].values 
            Y = df_subject.loc[:,['Score']].values
            REG = linear_model.LinearRegression()
            REG.fit(Y,X)

            st.write('Regression coefficient:', REG.coef_)
            st.write('Intercept:', REG.intercept_)

            plt.figure(figsize = (8,5))

            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.scatter(X,Y)
            plt.plot(Y, REG.predict(Y))
            plt.grid(True)
            plt.xlabel('Learning time')
            plt.ylabel('Score')
            st.pyplot()

            st.write('Coefficient of determination:',REG.score(Y,X))
            st.write('Correlation coefficient / P value:',sp.stats.pearsonr(df_subject['Learning time'],df_subject['Score']))

    except Exception as e:
        st.header('ERROR: Data inconsistency. Check data format to be uploaded.')
        print('Data inconsistency error')

