import streamlit as st
import train
import test
from os.path import exists
import os

st.set_page_config (page_title='PredictAAP', page_icon="icon1.png",layout='wide')



col1, col2, col3 = st.columns(3)

with col1:
    st.image("AICTE.png",width=200 )

with col2:
    st.title("Prediction of Admission And Placement",)
   

with col3:
    st.image("sih.png",width=180)

st.write('---')

with st.container():
     left_column, right_column = st.columns(2)

if exists("requirements.txt"):
    os.system("pip3 install -r requirements.txt")
    os.remove("requirements.txt")


if exists("model_sgd.h5"):
    st.markdown("<h1 style='text-align: center;'>Model Exists,Can Use the existing model for prediction </h1>", unsafe_allow_html=True)   
       
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
st.header("Select your choice")
choice=st.radio("",("Train the model","Test the model"))
if choice == 'Train the model':
    st.markdown("<h1 style='text-align: center; color:gray'>Training Section</h1>", unsafe_allow_html=True)
    train.train()
elif choice == 'Test the model':
    st.markdown("<h1 style='text-align: center; color:gray'>Testing Section</h1>", unsafe_allow_html=True)
    if exists("model_sgd.h5"):
        test.test()
    else:
        st.markdown("<h1 style='text-align: center;'>No Model Have been Trained please train the model first</h1>", unsafe_allow_html=True)
            
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 3rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)
         
    


