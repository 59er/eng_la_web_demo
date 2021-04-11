import streamlit as st
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib 
from sklearn.model_selection import train_test_split
import pickle
import shap 
from sklearn.ensemble import RandomForestClassifier
import base64

def run_edu_pass_fail_prediction_app():

    st.header('■Pass/Fail Prediction Demo')
    st.write('To know what makes students pass or fail.')

    st.sidebar.subheader('Data Upload')
    
    df_edu = pd.read_csv("data/eng_sample_pass_fail.csv")
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False, encoding = 'utf_8_sig')
            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    tmp_download_link = download_link(df_edu, 'sample_pass_fail.csv', 'Download sample csv file.')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)
    
#     st.sidebar.info("""
#     [Download the sample csv file](https://github.com/59er/eng_learning_analytics_web/blob/master/sample_data/eng_sample_pass_fail_for_WEB.csv)
#         """)
    uploaded_file = st.sidebar.file_uploader("File upload (Drag and drop or use [Browse files] button to import csv file. Only utf-8 format is available.）", type = ['csv'])

    try:

        if uploaded_file is not None:
            df_edu = pd.read_csv(uploaded_file)
            display_data = st.sidebar.checkbox(label = 'Show uploaded data')
            
            if display_data:
                st.dataframe(df_edu)

            df = df_edu.drop(['ID','Teacher'],axis=1)
            target = 'Pass/Fail'
            encode = ['Class','Subject']

            for col in encode:
                dummy = pd.get_dummies(df[col], prefix = col)
                df = pd.concat([df,dummy], axis = 1)
                del df[col]

            target_mapper = {'Fail':0, 'Pass':1}
            def target_encode(val):
                return target_mapper[val]
            
            df['Pass/Fail'] = df['Pass/Fail'].apply(target_encode)
            X = df.drop(['Pass/Fail'],axis=1)
            Y = df['Pass/Fail']
            clf = RandomForestClassifier(n_estimators = 250, random_state = 0)
            clf.fit(X, Y)
            df_drop = df.drop(['Pass/Fail'], axis = 1)

            prediction = clf.predict(df_drop)
            st.subheader('Prediction result')
            score_assess = np.array(['Fail','Pass'])
            id = df_edu['ID']
            result = score_assess[prediction]
            id = pd.DataFrame(id)
            result = pd.DataFrame(result)
            pred_result = pd.concat([id,result],axis=1)
            pred_result = pred_result.rename(columns = {0:'Result'})
            st.dataframe(pred_result)
            # st.write(score_assess[prediction])

            X = df.drop(['Pass/Fail'],axis=1)
            Y = df['Pass/Fail']
            X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

            import sklearn.tree as tree 
            load_clf = tree.DecisionTreeClassifier(random_state=0,max_depth=3)
            model = load_clf.fit(X_train,y_train)

            from sklearn.metrics import accuracy_score
            from sklearn.metrics import roc_curve, auc, roc_auc_score
            from sklearn.metrics import confusion_matrix
            pred = load_clf.predict_proba(X_test)[:,1]

            y_pred = np.where(pred > 0.6, 1, 0)

            score = accuracy_score(y_pred, y_test)
            st.subheader('Prediction accuracy')
            st.write(score)

            auc_score = roc_auc_score(y_test, pred)
            st.subheader('AUC accuracy ')
            st.write(auc_score)

            features  = X_train.columns
            importances = load_clf.feature_importances_
            indices = np.argsort(importances)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.figure(figsize = (6,6))
            plt.barh(range(len(indices)), importances[indices], color = '#377eb8', align = 'center')
            plt.yticks(range(len(indices)), features[indices])
            plt.show()
            st.pyplot()

        #confusion matrix
            matrix = confusion_matrix(y_pred, y_test)
            class_names = ['Fail','Pass']
            df = pd.DataFrame(matrix, index=class_names, columns=class_names)
            fig = plt.figure(figsize = (5,5))
            sns.heatmap(df, annot = True, cbar=None, cmap = 'Blues')
            plt.title('Prediction result')
            plt.tight_layout()
            plt.ylabel('Positive')
            plt.xlabel('Prediction')
            plt.show()
            st.pyplot(fig)

            from dtreeviz.trees import dtreeviz
            import graphviz as graphviz
            import streamlit.components.v1 as components

            viz = dtreeviz(
                model,
                X_train,
                y_train,
                target_name = 'Fail/Pass',
                feature_names = X_train.columns,
                class_names = ['Fail','Pass']
            )

            def st_dtree(plot, height = None):
                dtree_html = f'<body>{viz.svg()}</body>'
                components.html(dtree_html, height = height)

            st_dtree(dtreeviz(model, X_train, y_train,
                    target_name = 'Pass/Fail', feature_names = X_train.columns,
                    class_names = ['Fail','Pass']),400)


        else:
            def user_input_features():
                class_name = st.sidebar.selectbox('Class',('A','B','C','D','E','F','G'))
                subject = st.sidebar.selectbox('Subject',('History','Math','Literature'))
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

        sample_data = pd.read_csv('data/eng_sample_pass_fail.csv')
        sample = sample_data.drop(columns = ['Pass/Fail','ID','Teacher'])
        #df = sample.copy()
        df = pd.concat([input_df,sample], axis=0)
        # st.dataframe(df[:1])

        encode = ['Class','Subject']
        for col in encode:
            dummy = pd.get_dummies(df[col], prefix = col)
            df = pd.concat([df,dummy], axis = 1)
            del df[col]
        df1 = df[:1]
        # st.subheader('入力データ')

        # if uploaded_file is not None:
        #     st.write(df1)
        # else:
        st.write('The following is default sample data. Select class and subject then use the sliders for each item in the sidebar to get an idea of pass/fail prediction.')
        st.write(df1)


        load_clf = pickle.load(open('data/subject_pass_fail.pkl', 'rb'))
        prediction = load_clf.predict(df1)

        st.subheader('Prediction result')
        score_assess = np.array(['Fail','Pass'])
        st.write(score_assess[prediction])


        df1 = sample_data.copy()
        encode = ['Class','Subject']
        for col in encode:
            dummy1 = pd.get_dummies(df1[col], prefix = col)
            df1 = pd.concat([df1,dummy1], axis = 1)
            del df1[col]

        target_mapper = {'Fail':0, 'Pass':1}
        def target_encode(val):
            return target_mapper[val]

        df1['Pass/Fail'] = df1['Pass/Fail'].apply(target_encode)

        X = df1.drop(['Pass/Fail','Teacher','ID'],axis=1)
        Y = df1['Pass/Fail']
        X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

        # X1 = df1.drop(['得点'],axis=1)
        # prediction = load_clf.predict(X1)
        # st.subheader('得点予測結果')
        # st.dataframe(prediction)

        import sklearn.tree as tree 
        load_clf = tree.DecisionTreeClassifier(random_state=0,max_depth=3)
        model = load_clf.fit(X_train,y_train)

        from sklearn.metrics import accuracy_score
        from sklearn.metrics import roc_curve, auc, roc_auc_score
        from sklearn.metrics import confusion_matrix
        pred = load_clf.predict_proba(X_test)[:,1]

        y_pred = np.where(pred > 0.6, 1, 0)

        score = accuracy_score(y_pred, y_test)
        st.subheader('Prediction accuracy')
        st.write(score)

        auc_score = roc_auc_score(y_test, pred)
        st.subheader('AUC accuracy ')
        st.write(auc_score)

        features  = X_train.columns
        importances = load_clf.feature_importances_
        indices = np.argsort(importances)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize = (6,6))
        plt.barh(range(len(indices)), importances[indices], color = '#377eb8', align = 'center')
        plt.yticks(range(len(indices)), features[indices])
        plt.show()
        st.pyplot()

        #confusion matrix
        matrix = confusion_matrix(y_pred, y_test)
        class_names = ['Fail','Pass']
        df = pd.DataFrame(matrix, index=class_names, columns=class_names)
        fig = plt.figure(figsize = (5,5))
        sns.heatmap(df, annot = True, cbar=None, cmap = 'Blues')
        plt.title('Prediction result')
        plt.tight_layout()
        plt.ylabel('Positive')
        plt.xlabel('Prediction')
        plt.show()
        st.pyplot(fig)

        fig = plt.figure(figsize = (5,5))
        explainer = shap.TreeExplainer(load_clf,X)
        shap_values = explainer.shap_values(X)

        st.subheader('Impact of explanatory variables (each item score) on\
             the objective variable (final score): Class 0 has an impact for Fail, Class 1 for Pass)')
        fig = shap.summary_plot(shap_values, X , plot_type = 'bar')
        st.pyplot(fig)

        from dtreeviz.trees import dtreeviz
        import graphviz as graphviz
        import streamlit.components.v1 as components

        viz = dtreeviz(
            model,
            X_train,
            y_train,
            target_name = 'Fail/Pass',
            feature_names = X_train.columns,
            class_names = ['Fail','Pass']
        )

        def st_dtree(plot, height = None):
            dtree_html = f'<body>{viz.svg()}</body>'
            components.html(dtree_html, height = height)

        st_dtree(dtreeviz(model, X_train, y_train,
                target_name = 'Pass/Fail', feature_names = X_train.columns,
                class_names = ['Fail','Pass']),400)

    except Exception as e:
        print(e)
