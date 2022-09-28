import streamlit as st
import pandas as pd
from Models.model_sgd import default_model_training,custom_model_training
def train():
    with st.container():
     left_column, right_column = st.columns(2)
     with left_column:
      st.header("Choose CSV file")
      uploaded_file = st.file_uploader("")
    def create_file():
        data=create_list_frame(uploaded_file)
        if 'dummy_data' not in st.session_state.keys():
            dummy_data =data
            st.session_state['dummy_data'] = dummy_data
        else:
            dummy_data = st.session_state['dummy_data']
        return dummy_data



    #Create a list of columns of csv files and return rhe list and its dataframe
    def create_list_frame(uploaded_file):
        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)
            for _ in dataframe.columns:
                data=dataframe.columns.tolist()
            return data,dataframe



    #Checkbox container
    def checkbox_container(data):  
        for i in data:
            st.checkbox(i, key='dynamic_checkbox_' + i)

    #Get selected Checkbox
    def get_selected_checkboxes():
        return [i.replace('dynamic_checkbox_','') for i in st.session_state.keys()
        if i.startswith('dynamic_checkbox_') and st.session_state[i]]

    if uploaded_file is not None:    
        st.header('Select the type of features')
        c1=st.radio('',('Custom features','Default features'))
        if c1=='Custom features':
            try:
                data,dataframe=create_list_frame(uploaded_file)
                st.header('Select features\n')
                checkbox_container(data)
                c=get_selected_checkboxes()
                if "Admitted" not in c:
                    st.header("Please select the checkbox of admitted for training")
                else:
                    df=show_df=dataframe[c]
                    st.dataframe(show_df)
                    with open("features.txt","w") as f:
                        for i in c:
                            f.write(f"{i},")
                    if 'Sr. No.' in df.columns :
                        df.drop(['Sr. No.'],axis=1,inplace=True)
                    if 'Candidate Name' in df.columns:
                        df.drop(['Candidate Name'],axis=1,inplace=True)

                    if st.button("Train the Model"):
                        custom_model_training(df, st)
            except TypeError or NameError:
                    st.write("Please Select CSV File")
        if c1=='Default features':
            data,dataframe=create_list_frame(uploaded_file)
            c=dataframe[['District','Gender','SSC Math Percentage','SSC Total Percentage','HSC Physics Percentage','HSC Chemistry Percentage','HSC Math Percentage','HSC Subject Percentage','HSC English Percentage','HSC Total Percentage','CET Percentile','JEE Percentile','Merit No','Merit Marks','Admitted']]
            st.dataframe(c)
            with open("features.txt","w") as f:
                    for i in c:
                        f.write(f"{i},")
            if 'Sr. No.' in c.columns :
                c.drop(['Sr. No.'],axis=1,inplace=True)
            if 'Candidate Name' in c.columns:
                c.drop(['Candidate Name'],axis=1,inplace=True)
            
            if st.button("Train the Model"):
                default_model_training(c, st)
    else:
        st.header("Please Select CSV file")