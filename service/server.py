import streamlit as st
import joblib
import os
CLASS_NAMES = ['Beleing','Malig']
st.title("Project george brown Health Care")
st.markdown("Fill the information about the Cancer")
size = st.number_input("Enter the tumor size:")
hight= st.number_input("Enter the tumor hight:")
weith = st.number_input("Enter the tumor weith:")
submit = st.button('Predict')
if submit:
    model = joblib.load(os.environ.get("MODEL_PATH"))
    print(model)
    st.title(str(f"The cancer is{CLASS_NAMES[0]}"))
