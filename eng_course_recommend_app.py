# Core Pkg
import streamlit as st
import streamlit.components.v1 as stc
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer # Load EDA 
from sklearn.metrics.pairwise import linear_kernel
import base64

def run_course_recommend_app():

    st.header('■Course Recommend Demo')
    st.write('- To recommend optimal level courses for students.')
    st.write("In this demo, Udemy's course list is used as the source data. For example, if you enter 'How to make HTML file' in the Search field, the AI will recommend courses based on the analogy of the word.")


    st.sidebar.subheader('Data upload')
    
    df_edu = pd.read_csv("data/eng_sample_udemy_courses.csv")
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False, encoding = 'utf_8_sig')
            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    tmp_download_link = download_link(df_edu, 'sample_recommend.csv', 'Download sample csv file.')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)
    
#     st.sidebar.info("""
#     [Download the sample csv file](https://github.com/59er/eng_learning_analytics_web/blob/master/sample_data/eng_sample_udemy_courses_for_WEB.csv)

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

        else:
            df_edu = pd.read_csv("data/eng_sample_udemy_courses.csv")
            show_df = st.sidebar.checkbox('Show DataFrame')

            if show_df == True:
                st.write(df_edu)
        
        # @st.cache
        def search_term_if_not_found(term, df):
            result_df = df[df['course_title'].str.contains(term)]
            result_df = result_df.head()
            return result_df

        # st.subheader('Course recommend system')

        # menu = ['Course recommend','Original Data']
        # choice = st.selectbox('Menu',menu)
        df = df_edu.drop(['content_duration','subject'],axis=1)

        # if choice == 'Original Data':
        #     st.subheader('■Original Data')
        #     st.dataframe(df_edu.head(7))
        
        # elif choice == 'Course recommend':
        st.subheader('■Recommend courses')
        search_term = st.text_input('Search')
        if st.button('Recommend'):
            if search_term is not None:

                vectorizer = CountVectorizer()#vectorize model
                transformer = TfidfTransformer()#tfid model
                tf = vectorizer.fit_transform(df_edu['course_title'])
                tfid = transformer.fit_transform(tf)
                cosine_sim_mat = cosine_similarity(tfid)

                search_tf = vectorizer.transform([search_term])
                search_tfidf = transformer.transform(search_tf)#tfidf for test in search
                similarity = cosine_similarity(search_tfidf, tfid)[0] 
                topn_indices = np.argsort(similarity)[::-1][:7]

                _ = []
                for sim, outline in zip(similarity[topn_indices], np.array(df['course_title'])[topn_indices]):
                    _.append([sim, outline])
                #     print("({:.2f}): {}".format(sim, " ".join(outline.split())))
                recommend_course_list = pd.DataFrame(_)
                recommend_course_list = recommend_course_list.rename(columns = {0:'Similality Score', 1:'Course Title'})
                recommend_course_list['Similality Score'] = recommend_course_list['Similality Score'].round(2)
                recommend_course_list = recommend_course_list.reindex(columns =['Course Title', 'Similality Score'])
                if (recommend_course_list['Similality Score'] == 0).sum() == recommend_course_list.shape[0]:
                    results = 'Not found'
                    st.warning(results)
                    st.info('Suggested Options include')
                    result_df = search_term_if_not_found(search_term, df)
                    st.dataframe(result_df)
                else:
                    st.dataframe(recommend_course_list)
                    recommend_course_list.to_csv('data/recommend_result.csv')
                    recommend_result = pd.read_csv("data/recommend_result.csv")

                    def download_link(object_to_download, download_filename, download_link_text):
                        if isinstance(object_to_download,pd.DataFrame):
                            object_to_download = object_to_download.to_csv(index=False)
                            b64 = base64.b64encode(object_to_download.encode('utf-8-sig')).decode()
                            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

                    tmp_download_link = download_link(recommend_result, 'recommend_result.csv', 'Download the csv of the displayed cource recommend.')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)

    except Exception as e:
        st.header('ERROR: Data inconsistency. Check data format to be uploaded.')
        print('Data inconsistency error')
