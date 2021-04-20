import streamlit as st
import streamlit.components.v1 as stc 
from eng_edu_overview_app import run_edu_overview_app
from eng_edu_detail_app import run_edu_detail_app
from eng_test_cronbach_alpha_app import run_test_cronbach_alpha_app
from eng_edu_score_prediction_app import run_edu_score_prediction_app
from eng_edu_pass_fail_prediction_app import run_edu_pass_fail_prediction_app
from eng_cluster_app import run_cluster_app
from eng_corr_app import run_corr_app
from eng_text_mining_app import run_text_mining_app
from eng_test_difficulty_app import run_test_difficulty_app
from eng_t_test_app import run_t_test_app
from eng_factor_app import run_factor_app
from eng_time_analysis_app import run_time_analysis_app
from eng_edu_score_prediction_app import run_edu_score_prediction_app
from eng_course_recommend_app import run_course_recommend_app
from eng_sampling_adequacy_app import run_sampling_adequacy_app

def main():
    st.title("Learning Analytics WEB Demo\n (β version)")

    menu = ["HOME",'Overview of learning situation','Detailed analysis',
            'Correlation analysis',
            'Score prediction',
            'Pass/Fail Prediction','Questionnaire text analysis',
            'Exam questions/questionnaire reliability measurement','Exam questions difficulty and discrimination analysis','Sampling adequacy',
            'Cluster analysis','t-test','Factor analysis','Time series analysis','Course recommend']
 
    choice = st.sidebar.selectbox("MENU",menu)
    st.sidebar.text('Select a function from the MENU.')

    # choice = st.sidebar.selectbox("Menu",menu)

    if choice =='HOME':
        st.header("■Analytic Function Menu")
        st.write('The following learning analysis can be selected from the sidebar menu.')
        # st.subheader('Menu')
        st.subheader('- Overview of learning situation:')
        st.write(' To get an overview of the test results. To understand the trend of each class visually.')
        st.subheader('- Detailed analysis: ')
        st.write('To investigate the relationship between learning time and academic achievement.')
        st.subheader('- Correlation analysis: ')
        st.write('To investigate the relationship between midterm and final exam grades.')
        st.subheader('- Score prediction: ')
        st.write(' To predict the expected score of students who are absent from the test.')
        st.subheader('- Pass/Fail Prediction: ')
        st.write('To know what makes students pass or fail.')
        st.subheader('- Questionnaire text analysis:')
        st.write(' To analyze the free text of the class evaluation questionnaire.')
        st.subheader('- Exam questions/questionnaire reliability measurement: ')
        st.write('To measure the validity and reliability of tests.')
        st.subheader('- Difficulty and discrimination analysis for Exam Questions:')
        st.write(' To grade each test based on its difficulty and discrimination.')
        st.subheader('- Cluster analysis: ')
        st.write('To group students who have similar learning characteristics.')
        st.subheader('- t-test: ')
        st.write('To compare the results of two tests. e.g., examine the difference in performance by teaching method.')
        st.subheader('Sampling adequacy')
        st.write('To investigate the adequay of the number of samples for questionnaire.')
        st.subheader('- Factor analysis: ')
        st.write('To create and analyze a class evaluation questionnaire.')
        st.subheader("- Time series analysis:")
        st.write( "To visualize the academic achievement over time.")
        st.subheader("- Course recommend")
        st.write("To recommend optimal study course for learners")
        
    elif choice == "Overview of learning situation":
        # run_overview_app()
        run_edu_overview_app()

    elif choice == "Detailed analysis":
        run_edu_detail_app()

    elif choice == "Correlation analysis":
        run_corr_app()

    elif choice == "Score prediction":
        run_edu_score_prediction_app()

    elif choice == "Pass/Fail Prediction":
        run_edu_pass_fail_prediction_app()

    elif choice == "Questionnaire text analysis":
        run_text_mining_app()

    elif choice == "Exam questions/questionnaire reliability measurement":
        run_test_cronbach_alpha_app()

    elif choice == 'Exam questions difficulty and discrimination analysis':
        run_test_difficulty_app()
        
    elif choice == "Sampling adequacy":
        run_sampling_adequacy_app()

    elif choice == "Cluster analysis":
        run_cluster_app()

    elif choice == "t-test":
        run_t_test_app()

    elif choice == "Factor analysis":
        run_factor_app()

    elif choice == 'Time series analysis':
        run_time_analysis_app()

    elif choice == 'Course recommend':
        run_course_recommend_app()

if __name__ == '__main__':
    main()
