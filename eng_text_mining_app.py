import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sus 
from matplotlib.font_manager import FontProperties
import nlplot
import streamlit as st
from PIL import Image, ImageDraw
import base64

def run_text_mining_app():
    st.header('■Questionnaire text analysis')
    st.write('To analyze the free text of the class evaluation questionnaire. You can try Word Count, Tree Map and Sunburn Chart from the item menu below. ')
    
    st.sidebar.subheader('Data Upload')

    df_edu = pd.read_csv("data/eng_sample_low_score_words.csv")
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False, encoding = 'utf_8_sig')
            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    tmp_download_link = download_link(df_edu, 'sample_text_mining.csv', 'Download sample csv file.')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)

#     st.sidebar.info("""
#     [Download the sample csv file](https://github.com/59er/eng_learning_analytics_web/blob/master/sample_data/eng_sample_low_score_words_for_WEB.csv)
#         """)

    uploaded_file = st.sidebar.file_uploader("File upload (Drag and drop or use [Browse files] button to import csv file. Only utf-8 format is available.）", type=["csv"])
	# uploaded_file = st.file_uploader(
	#     label = 'File Upload（Drag and drop csv/Excel）',
	#     type = ['csv', 'xlsx']
	# )

    try:

        if uploaded_file is not None:
            df_edu = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            display_data = st.sidebar.checkbox(label = 'Show uploaded data')
            
            if display_data:
                st.dataframe(df_edu)

            npt = nlplot.NLPlot(df_edu, target_col = 'comments')

            stopwords = npt.get_stopword(top_n = 1, min_freq = 1)

            item_select = st.selectbox(
            label = 'Select item',
            options = ['Word count','Tree Map',
                        'Sunburst Chart'])

            if item_select == 'Word count':
                st.write(npt.bar_ngram(title='Word count(one word)', ngram = 1, top_n = 20, stopwords = stopwords, width = 800, height = 700))
                st.write(npt.bar_ngram(title='Word count(two word)', ngram = 2, top_n = 20, stopwords = stopwords, width = 800, height = 700))

            if item_select == 'Tree Map':
                st.write(npt.treemap(title='Tree Map', ngram=1, top_n = 20, stopwords = stopwords, width = 800, height = 700))


            #if item_select == 'Co-occurrence network':
            #    npt.build_graph(stopwords = stopwords, min_edge_frequency = 1)
            #    st.write(npt.co_network(title = 'Co-occurrence network',width = 800, height = 700))

            if item_select == 'Sunburst Chart':
                npt.build_graph(stopwords = stopwords, min_edge_frequency=1)
                st.write(npt.sunburst(title='Sunburst Chart', width = 800, height=700))


#             if item_select == 'Word Cloud':
#                 st.set_option('deprecation.showPyplotGlobalUse', False)
#                 npt.build_graph(stopwords = stopwords, min_edge_frequency=2)
#                 st.write(npt.wordcloud(max_words = 100, max_font_size = 100, stopwords = stopwords, 
#                 colormap = 'tab20_r', width = 500, height = 400))
#                 st.pyplot()
            

        else:
            df_edu = pd.read_csv('data/eng_sample_low_score_words.csv')
            show_df = st.sidebar.checkbox('Show dataFrame')

            if show_df == True:
                st.write(df_edu)

            npt = nlplot.NLPlot(df_edu, target_col = 'comments')

            stopwords = npt.get_stopword(top_n = 1, min_freq = 1)

            item_select = st.selectbox(
            label = 'Select item',
            options = ['Word count','Tree Map',
                        'Sunburst Chart'])

            if item_select == 'Word count':
                st.write(npt.bar_ngram(title='Word count(one word)', ngram = 1, top_n = 20, stopwords = stopwords, width = 800, height = 700))
                st.write(npt.bar_ngram(title='Word count(two word)', ngram = 2, top_n = 20, stopwords = stopwords, width = 800, height = 700))

            if item_select == 'Tree Map':
                st.write(npt.treemap(title='Tree Map', ngram=1, top_n = 20, stopwords = stopwords, width = 800, height = 700))


#             if item_select == 'Co-occurrence network':
#                 npt.build_graph(stopwords = stopwords, min_edge_frequency = 1)
#                 st.write(npt.co_network(title = 'Co-occurrence network',width = 800, height = 700))

            if item_select == 'Sunburst Chart':
                npt.build_graph(stopwords = stopwords, min_edge_frequency=1)
                st.write(npt.sunburst(title='Sunburst Chart', width = 800, height=700))


#             if item_select == 'Word Cloud':
#                 st.set_option('deprecation.showPyplotGlobalUse', False)
#                 npt.build_graph(stopwords = stopwords, min_edge_frequency=2)
#                 st.write(npt.wordcloud(max_words = 100, max_font_size = 100, stopwords = stopwords, 
#                 colormap = 'tab20_r', width = 500, height = 400))
#                 st.pyplot()
            
    except Exception as e:
        st.header('ERROR: Data inconsistency. Check data to be uploaded.')
        print('Data inconsistency error')
