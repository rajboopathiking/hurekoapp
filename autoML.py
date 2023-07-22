#!/usr/bin/env python
# coding: utf-8

# In[18]:


import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport


# In[20]:


from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import *


# In[22]:
df = pd.read_csv("automl.csv",index_col=None)

st.title("Auto Mechine Learning")
st.image("/home/datahagward/Downloads/3d-robot-hand-background-ai-technology-side-view.jpg")

with st.sidebar:
    st.title("Classification model")
    option = st.radio("navigation",["upload","profiling","model","download"])
    
if option == "upload":
    file = st.file_uploader("upload_here")
    if file:
       df = pd.read_csv(file,index_col=None)
       df.to_csv("automl.csv",index=None)
       st.dataframe(df)

if option == "profiling":

    report = ProfileReport(df)
    st_profile_report(report)

if option == "model":
    target = st.selectbox("select_columns",[col for col in df.columns if df[col].dtypes != "O"])
    from imblearn.over_sampling import RandomOverSampler
    

    ros = RandomOverSampler(random_state=42)
    df, target = ros.fit_resample(df, df[target])
    
    setup(df,target=target)
    setup_df = pull()
    st.dataframe(setup_df)
    best_model = compare_models()
    compare_models_df = pull()
    st.dataframe(compare_models_df)
    save_model(best_model,"automl")

if option == "download":

    with open("automl.pkl","rb") as file:
       st.download_button("download_model",file,"automl.pkl")

