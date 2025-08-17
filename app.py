import streamlit as st 
import pandas as pd
import numpy as np
import tensorflow as tf
from pandas._libs.interval import Interval
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

model = tf.keras.models.load_model('model.h5')

with open('one_hot_encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('standard_scaler.pkl', 'rb') as file:
    standard_scaler = pickle.load(file)
    
### for the streamlit webapp
st.title("Consumer Churn Predictions")

# user input for the web apps
geography = st.selectbox("Geography",one_hot_encoder.categories_[0])
gender = st.selectbox("Gender",label_encoder.classes_)
age = st.slider("Age",18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('tenure',0,10)
num_of_products = st.slider('Number of Products',1,4,1)
has_cr_cards = st.selectbox('Has Credit Card',[0,1])
active_member = st.selectbox('Is Active Member',[0,1])


input_data = {
    'CreditScore':credit_score,
    'Gender':label_encoder.transform([gender])[0],
    'Age':age,
    'Tenure': tenure,
    'Balance':balance,
    'NumOfProducts':num_of_products,
    'HasCrCard':has_cr_cards,
    'IsActiveMember':active_member,
    'EstimatedSalary':estimated_salary
}

input_df = pd.DataFrame([input_data])


encoded_geo = one_hot_encoder.transform([geography]).toarray()
encoded_df = pd.DataFrame(encoded_geo, columns=one_hot_encoder.get_feature_names_out(['Geography'])) 
input_df = pd.concat([input_df.reset_index(drop=True), encoded_df], axis=1)

input_scaled = standard_scaler.transform(input_df)
input_scaled = np.array(input_scaled).reshape(1, -1)

prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]
st.write(f"Prediction Probability: {prediction_prob:.2f}")

if prediction_prob >0.5:
    st.write("the consumer is most likely to be churned")
else:
    st.write("this consumer is not likely to be churned")
    

    



