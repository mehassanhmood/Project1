import streamlit as st
import joblib
import os
from streamlit_modal import Modal
import hydralit_components as hc
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost
hc.hydralit_experimental(True)
CLASS_NAMES = ['No Stroke','Stroke']
st.title("Project George Brown Health Care Stroke Classification")
st.markdown("Fill your health attributes.")
age = st.number_input("Enter your age:")
#gender= st.selectbox(
 #   'Select your gender?',
  #  ('Male','Female'))
hypertension = st.selectbox(
    'Do you have hypertension?',
    ('Yes','No'))
heart_disease = st.selectbox(
    'Do you have Heart Disease?',
    ('Yes','No'))
smoke_status = st.selectbox(
    'Do you smoke?',
    ('Never','Yes',"Sometimes"))
glucose_level= st.number_input("Enter your latest blood glucose level:")
#bmi = st.number_input("Enter with your Bmi")
residence = st.selectbox(
    'Area of residence?',
    ('Urban','Rural'))
work = st.selectbox(
    'What kind of work are you in ?',
    ('Self-employed ','Private',"Public"))

modal = Modal(title="Data Permission",key="modal-data-sensitive",max_width=450)
modal_code = """
<div>
<!-- Button trigger modal -->
<button type="button" class="btn btn-primary" data-toggle="modal" data-target="#exampleModal">
Submit
</button>

<!-- Modal -->
<div class="modal fade" id="exampleModal" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel" aria-hidden="true">
<div class="modal-dialog" role="document">
<div class="modal-content">
<div class="modal-header">
  <h5 class="modal-title" id="exampleModalLabel" style="color:black">Data Permission</h5>
  <button type="button" class="close" data-dismiss="modal" aria-label="Close">
    <span aria-hidden="true">&times;</span>
  </button>
</div>
<div class="modal-body">
  <div class="container">
<h4 style="color:white">Do you consent for your data to be stored and shared with health care providers. ? </h4>
<form class="form-horizontal" action="/">
            <div class="form-group">
                <label for="choice">Choose:</label>
                <select class="form-control" id="choice" name="choice">
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                </select>
            </div>
            <button type="submit" class="btn btn-primary">Submit</button>
    </form>
</div>
</div>
<div class="modal-footer">
  <button type="button" id="addQueryParamButton"  class="btn btn-secondary" data-dismiss="modal">Yes</button>
  <button type="button" class="btn btn-primary" ata-dismiss="modal">No</button>
</div>
</div>
</div>
</div>
</div>
<script>
    // Function to add query parameters to the URL
    function addQueryParam(key, value) {
        // Get the current URL
        var currentUrl = window.location.href;

        // Check if the URL already contains a question mark (?)
        var separator = currentUrl.includes('?') ? '&' : '?';

        // Construct the new URL with the query parameter
        var newUrl = currentUrl + separator + key + '=' + value;
    // Update the URL
        window.history.replaceState({}, '', newUrl);
    }

    // Attach a click event listener to the button
    var addButton = document.getElementById('addQueryParamButton');
    addButton.addEventListener('click', function () {
        // Call the addQueryParam function with the desired key and value
        addQueryParam('process', 'True');
    });
</script>
"""


st.markdown(modal_code,unsafe_allow_html=True)
query_param = st.experimental_get_query_params()

print(query_param)
if "choice" in query_param:
    if query_param["choice"][0] == "yes":
        # Create a dictionary with the user inputs
            data = {
                'age': [age],
                'hypertension': [1 if hypertension == 'Yes' else 0],
                'heart_disease': [1 if heart_disease == 'Yes' else 0],
                'avg_glucose_level': [glucose_level],
                'bmi': [22],
                'gender_Male': [1 if 1 == 'Male' else 0],
                #'ever_married_Yes': [0],  # Assuming 'ever_married' is not taken as input
                #'work_type_Never_worked': [0],  # Assuming work_type is not taken as input
                #'work_type_Private': [0],  # Assuming work_type is not taken as input
                #'work_type_Self-employed': [1 if work== "Self-employed" else 0],  # Assuming work_type is not taken as input
                #'work_type_children': [0],  # Assuming work_type is not taken as input
                #'Residence_type_Urban': [1 if residence == 'Urban' else 0 ],  # Assuming Residence_type is not taken as input
                'smoking_status_formerly_smoked': [1 if smoke_status == 'Sometimes' else 0],
                'smoking_status_never_smoked': [1 if smoke_status == 'Never' else 0],
                'smoking_status_smokes': [1 if smoke_status == 'Yes' else 0]
                }

            # Create a DataFrame
            # st.markdown(os.environ.get('DATASET_PATH'))
            df = pd.DataFrame(data)
            # df2 = pd.read_csv(os.environ.get("DATASET_PATH"))
            # df2 = pd.read_csv('./data/healthcare-dataset-stroke-data-clean.csv')[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_Male', 'smoking_status_formerly_smoked', 'smoking_status_never_smoked', 'smoking_status_smokes']]
            # df2 = pd.concat([df2,df]).reset_index(drop = True)
            scaled = [StandardScaler().fit_transform(df)]
            # model = joblib.load(os.environ.get("MODEL_PATH"))
            model = joblib.load('../xgb_model1.pkl')
            pred = model.predict(scaled)
            predict=model.predict_proba(scaled)
            proba = "{:.2f}".format(predict[0][1]*100)
            print(predict)
            #st.markdown(predict)
            st.title(str(f"Based onthe provided data following are the results: {proba}% at risk of stroke"))
                # Add your flow code here 
