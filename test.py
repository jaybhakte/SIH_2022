import streamlit as st
import pandas as pd
from Models.model_sgd import replace_categorical,replace_nan_with_zeros
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
def test():
    try:
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
    
        if uploaded_file is not None:    
            data,df=create_list_frame(uploaded_file)
            features=[]
            with open("features.txt","r") as f:
                features.append(f.readlines()) 
            final_features=str(features[0])[2:-3].split(",")
            if "Admitted" in final_features:
                final_features.remove("Admitted")
            if "Sr. No." in final_features:
                final_features.remove("Sr. No.")
            try:
                st.dataframe(df[final_features])
            except KeyError:
                for n,m in enumerate(final_features):
                    if m=="Admitted":
                        final_features.remove(m)
                    if m=="JEE Percentile":
                        final_features[n]="JEE Score"
                    if m=="CET Percentile":
                        final_features[n]="CET Score"
                    if m=="JEE Total Score":
                        final_features[n]="JEE Score" 
                    if m=="CET Total Score":
                        final_features[n]="CET Score" 
            df=df[final_features]
            st.write("Scaling the data...")
            df_cpy=df
            
            
            sections =[i for i in df.columns if df[i].dtype==object]
            #st.write(sections)
            
            
            
            #Replace categorical here only
            df=replace_nan_with_zeros(df)
            df=replace_categorical(df)
            scaler = MinMaxScaler()
            normalised = scaler.fit_transform(df)
            
            #Load the model
            model = load_model("model_sgd.h5")
            c=model.predict(normalised)
            c_df=[]
            for i in c:
                if i>0.60:
                    c_df.append(i)
            c_df=pd.DataFrame(c_df)
            st.markdown(f"<h5 style='text-align: center; color:grey'>Total number of students that are predicted to take admission are-:{len(c_df)} out of {len(c)}</h5>", unsafe_allow_html=True)
            st.header("Press on the button to generate report.")
            st.markdown("<h5 style='text-align: center; color:grey'>Note That The Reports will be generated only of the features you have been selected for model training</h5>", unsafe_allow_html=True)
            for i in sections:
                if st.checkbox(i):
                    st.markdown(f"<h1 style='text-align: center;'>Reports of {i}</h1>", unsafe_allow_html=True)
                    temp_df=df_cpy[i]
                    temp_df=temp_df.to_frame()
                    temp_df['Admitted']=pd.DataFrame(c_df)
                    temp_df=temp_df.dropna()
                    wanted_df=temp_df
                    final_df=wanted_df.groupby([i]).sum()
                    why_df = pd.DataFrame()
                    why_df["Admitted"]=final_df['Admitted'].apply(lambda x: round((x/final_df["Admitted"].sum())*100,2))
                    if i not in ["Gender","Region"]:
                        ax=why_df.plot.bar(y='Admitted',figsize=(8, 8),xlabel=f"{i}", ylabel="Percentages of admission",logy=True)
                    else:
                        ax=why_df.plot.bar(y='Admitted',figsize=(8, 8),xlabel=f"{i}", ylabel="Percentages of admission")
                    for p in ax.patches:
                        width = p.get_width()
                        height = p.get_height()
                        x, y = p.get_xy() 
                        ax.annotate(f'{height}%', (x + width/2, y + height*1.02), ha='center')
                    st.pyplot(plt.show())
                    plt.title("Prediction of admissions")
                    final_df.plot(kind='pie', y='Admitted',autopct='%1.0f%%',figsize=(8,8))
                    plt.pause(0.01)
                    st.pyplot(plt.show())
        else:
            st.header("Please Select CSV file")
    except ValueError as err:
        st.write(err)
        st.header("The Dataset contains the differnt columns which have not been used while training")
