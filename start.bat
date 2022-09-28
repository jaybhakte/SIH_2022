@ECHO off
IF EXIST requirements.txt pip3 install -r requirements.txt
IF EXIST del requirements.txt

streamlit run streamlit_app.py