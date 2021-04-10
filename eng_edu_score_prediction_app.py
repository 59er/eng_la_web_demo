import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import japanize_matplotlib 
from sklearn.model_selection import train_test_split
import pickle
import shap 
#shap.initjs
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor
import base64

def run_edu_score_prediction_app():

    st.header('■Score prediction Demo')
    st.write('To predict the expected score (e.g., for students who are absent from the test.)')

    st.sidebar.subheader('Data Upload')
    
    df_edu = pd.read_csv("data/eng_sample_data_score_prediction.csv")
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False, encoding = 'utf_8_sig')
            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    tmp_download_link = download_link(df_edu, 'sample_score_pred.csv', 'Download sample csv file.')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)
    
#     st.sidebar.info("""
#     [Download the sample csv file](https://github.com/59er/eng_learning_analytics_web/blob/master/sample_data/eng_sample_data_score_prediction_for_WEB.csv)
#         """)
    try:

        uploaded_file = st.sidebar.file_uploader("File upload (Drag and drop or use [Browse files] button to import csv file. Only utf-8 format is available.）", type=["csv"])

        if uploaded_file is not None:
            df_edu = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            display_data = st.sidebar.checkbox(label = 'Show uploaded data')
            
            if display_data:
                st.dataframe(df_edu)

            df = df_edu.drop(['ID','Teacher'],axis=1)
            target = 'Score'
            encode = ['Class','Subject']

            for col in encode:
                dummy = pd.get_dummies(df[col], prefix = col)
                df = pd.concat([df,dummy], axis = 1)
                del df[col]
            X = df.drop(['Score'],axis=1)
            Y = df['Score']
            X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

            clf = RandomForestRegressor()
            clf.fit(X, Y)
            df = df.drop(['Score'],axis=1)
            prediction = clf.predict(df)
            st.subheader('Score prediction result')
            id = df_edu['ID']
            id = pd.DataFrame(id)
            result = pd.DataFrame(prediction)
            pred_result = pd.concat([id,result],axis=1)
            pred_result = pred_result.rename(columns = {0:'Result'})
            st.dataframe(pred_result)

            score = clf.score(X_test, y_test)
            st.set_option('deprecation.showPyplotGlobalUse', False)

            st.subheader('Prediction accuracy')
            st.write(score)

            fig = plt.figure(figsize = (5,5))
            explainer = shap.TreeExplainer(clf,X)
            shap_values = explainer.shap_values(X)

            st.subheader('Impact of explanatory variables (each item score) on the objective variable (final score)')
            fig = shap.summary_plot(shap_values, X , plot_type = 'bar')
            st.pyplot(fig)

            st.subheader('Correlation of explanatory variables with the objective variable (final score)')
            fig1 = shap.summary_plot(shap_values, X)
            st.pyplot(fig1)

            def st_shap(plot, height=None):
                shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                components.html(shap_html, height=height)


        else:

            def user_input_features():
                class_name = st.sidebar.selectbox('Class',('A','B','C','D','E','F','G'))
                subject = st.sidebar.selectbox('Subject',('Literature','Math','Reading'))
                subject_A = st.sidebar.slider('item1',0,100,50)
                subject_B = st.sidebar.slider('item2',0,100,50)
                subject_C = st.sidebar.slider('item3',0,100,50)
                subject_D = st.sidebar.slider('item4',0,100,50)
                subject_E = st.sidebar.slider('item5',0,100,50)
                data = {'Class': class_name,
                        'Subject': subject,
                        'item1': subject_A,
                        'item2': subject_B,
                        'item3': subject_C,
                        'item4': subject_D,
                        'item5': subject_E
                        }
                features = pd.DataFrame(data, index=[0])
                return features
            input_df = user_input_features()

        sample_data = pd.read_csv('data/eng_sample_data_score_prediction.csv')
        sample = sample_data.drop(columns = ['Score','ID','Teacher'])
        #df = sample.copy()
        df = pd.concat([input_df,sample], axis=0)
        # st.dataframe(df[:1])

        encode = ['Class','Subject']

        for col in encode:
            dummy = pd.get_dummies(df[col], prefix = col)
            df = pd.concat([df,dummy], axis = 1)
            del df[col]
        df1 = df[:1]

        if uploaded_file is not None:
            st.write(df1)
        else:
            st.write('The following is default sample data. Select class and subject then use the sliders for each item in the sidebar to get an idea of score prediction.')
            st.write(df1)

        load_clf = pickle.load(open('data/subject_score_prediction.pkl', 'rb'))
        prediction = load_clf.predict(df1)
        st.subheader('Score prediction result')
        st.write(prediction[0])


        df1 = sample_data.copy()
        encode = ['Class','Subject']
        for col in encode:
            dummy1 = pd.get_dummies(df1[col], prefix = col)
            df1 = pd.concat([df1,dummy1], axis = 1)
            del df1[col]


        X = df1.drop(['Score','Teacher','ID'],axis=1)
        Y = df1['Score']
        X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)


        score = load_clf.score(X_test, y_test)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.subheader('Prediction accuracy')
        st.write(score)

        fig = plt.figure(figsize = (5,5))
        explainer = shap.TreeExplainer(load_clf,X)
        shap_values = explainer.shap_values(X)

        st.subheader('Impact of explanatory variables (each item score, class and subject) on the objective variable (final score)')
        fig = shap.summary_plot(shap_values, X , plot_type = 'bar')
        st.pyplot(fig)

        st.subheader('Correlation of explanatory variables (each item score, class and subject) with the objective variable (final score)')
        fig1 = shap.summary_plot(shap_values, X)
        st.pyplot(fig1)

        def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)

        #force_plot for some IDs
        # i = np.arange(1,5,1)
        # for i in i:
        #     st.write('Example of', 'ID', i)
        #     st_shap(shap.force_plot(explainer.expected_value, shap_values[i,:],X.iloc[i,:]),400)

        # st_shap(shap.force_plot(explainer.expected_value, shap_values, X),400)
        
    except Exception as e:
        print(e)
