import pickle
import streamlit as st
import numpy as np
with open('salary_model.pkl', 'rb') as f:
    model=pickle.load(f)
st.title("Salary Predicition System")
st.write ("enter years of experience")
experience=st.number_input("years of experience",min_value=0.0,max_value=100.0,step=0.1)
if st.button("Predict Salary"):
    prediction=model.predict(np.array([[experience]]))
    st.write(f"Predicted Salary: {prediction[0]:.2f}")
                             