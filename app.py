import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import numpy as np


model=tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb')as file:
    onehot_encoder_geo=pickle.load(file)

with open('scalar.pkl','rb')as file:
    scalar=pickle.load(file)


st.title('Customer Churn Prediction')

geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('CreditScore')
estimated_salary=st.number_input('EstimatedSalary')
num_of_products=st.slider('NumOfProducts', 1, 4)
tenure=st.slider('Tenure',0,10)
has_cr_card=st.selectbox('HasCrCard',[0,1])
is_active_member=st.selectbox('IsActiveMember',[0,1])

input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
    })

geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data=scalar.transform(input_data)

prediction=model.predict(input_data)
prediction_prob=prediction[0][0]

if prediction_prob > 0.5 :
    st.write('The Customer likely to churn')

else:
    st.write('The customer not likely to churn')