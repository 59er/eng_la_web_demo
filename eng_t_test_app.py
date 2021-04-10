import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import japanize_matplotlib
from scipy import stats 
import streamlit as st
import base64

def run_t_test_app():
    st.header('■t-test')
    st.write('To compare the results of two tests. e.g., examine the difference in performance by teaching method.')

    st.sidebar.subheader('Data Upload')

    df_edu = pd.read_csv("data/eng_sample_data_t_test.csv")
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False, encoding = 'utf_8_sig')
            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    tmp_download_link = download_link(df_edu, 'sample_ttest.csv', 'Download sample csv file.')
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)

#     st.sidebar.info("""
#     [Download the sample csv file](https://github.com/59er/eng_learning_analytics_web/blob/master/sample_data/eng_sample_data_t_test_for_WEB.csv)
#         """)

    uploaded_file = st.sidebar.file_uploader("File upload (Drag and drop or use [Browse files] button to import csv file. Only utf-8 format is available.)", type=["csv"])
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

            

        else:
            df_edu = pd.read_csv('data/eng_sample_data_t_test.csv')

            show_df = st.sidebar.checkbox('Show DataFrame')

            if show_df == True:
                st.write(df_edu) 


        A_var = np.var(df_edu.iloc[:,0], ddof = 1)
        B_var = np.var(df_edu.iloc[:,1], ddof = 1)
        A_df = len(df_edu) - 1
        B_df = len(df_edu) - 1
        f = A_var/B_var
        one_sided_pval1 = stats.f.cdf(f,A_df,B_df)
        one_sided_pval2 = stats.f.sf(f,A_df,B_df)
        two_sided_pval = min(one_sided_pval1, one_sided_pval2)

        st.subheader("Confirmation of equality of variance between two groups (p-value < 0.05 for unequal variance (Welch's t-test was applied)),\
            Equal variances at p-value > 0.05 (Student's t-test applied))")
        dist = round(two_sided_pval,3)
        st.write('F      ', round(f,3))
        st.write('p-value:', round(two_sided_pval,3))

        if dist < 0.05:

            result_w = stats.ttest_ind(df_edu.iloc[:,0], df_edu.iloc[:,1])
            st.subheader('t-test results (welch)')
            st.write(result_w)

        else:
            result_s = stats.ttest_ind(df_edu.iloc[:,0],df_edu.iloc[:,1])
            st.subheader('t-test results (Student)')
            st.write(result_s)

        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.write(sns.catplot(x = 'variable', y = 'value', kind = 'box',data = pd.melt(df_edu)))
        plt.title('Comparison between the two groups', fontsize = 15)
        plt.show()
        st.pyplot()

    except Exception as e:
        st.header('ERROR: Data inconsistency. Check data format to be uploaded.')
        print('Data inconsistency error')
