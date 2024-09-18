import pickle

import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from PIL import Image

model = pickle.load(open('model1234.sav','rb'))
scaler = pickle.load(open('scaler.sav','rb'))



st.title('Health Insurance Cost Prediction')
st.sidebar.header('Patient Data')
image = Image.open('imgg.jpg')
st.image(image)

def user_report():
    age = st.sidebar.slider('age',0,70,0)
    sex = st.sidebar.slider('sex',0,1,0)
    bmi = st.sidebar.slider('bmi',0,60,0)
    children = st.sidebar.slider('children',0,4,0)
    smoker = st.sidebar.slider('smoker',0,1,0)
    region = st.sidebar.slider('region',0,4,0)
    
    
    patient_report_data={
        'age': age,
        'sex':sex,
        'bmi':bmi,
        'children':children,
        'smoker':smoker,
        'region':region
        
    }
    
    report_data = pd.DataFrame(patient_report_data,index=[0])
    return report_data


user_data = user_report()
st.header('Patient Data')
st.write(user_data)


salary = model.predict(scaler.transform(user_data))
st.subheader('Cost Of Insurance')
st.subheader('$'+str(np.round(salary[0],2)))
    
    
    
    

    