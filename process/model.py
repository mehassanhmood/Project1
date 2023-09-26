import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
#from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, roc_curve ,auc
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import os
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import warnings
warnings.filterwarnings('ignore')
def extract_data(data_path):
    df = pd.read_csv(data_path)
    return df 

def transform_data(dataframe):
    dataframe.dropna(inplace=True)
    dataframe.drop(dataframe[(dataframe.gender!='Male') & (dataframe.gender != 'Female')].index,inplace=True)
    to_encode = dataframe.select_dtypes(include='object')
    encode_columns = to_encode.columns
    encode = OneHotEncoder(drop='first')
    encoded_df = encode.fit_transform(dataframe[encode_columns])
    encoded_df = pd.DataFrame(encoded_df.toarray(),columns=encode.get_feature_names_out(encode_columns))
    df_prep = pd.concat([dataframe.select_dtypes(exclude='object').reset_index(drop = True),encoded_df],axis = 'columns')
    df_prep.columns = ['id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level',
       'bmi', 'gender_Male', 'ever_married_Yes',
       'work_type_Never_worked', 'work_type_Private',
       'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban',
       'smoking_status_formerly smoked', 'smoking_status_never smoked',
       'smoking_status_smokes', 'stroke']
    df_prep = df_prep.drop(columns='id')
    return df_prep

def split_features_target(dataframe):
    x,y = dataframe.iloc[:,:-1],dataframe.iloc[:,-1]
    return x ,y
def split_train_test(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,stratify=y,random_state=10)
    training_set = pd.concat([X_train,y_train],axis=1)
    majority_class = training_set[training_set.stroke == 0]
    minority_class = training_set[training_set.stroke == 1]
    minority_upsampled = resample(minority_class,replace=True,n_samples=int(len(majority_class)*0.5),random_state=42)
    upsampled = pd.concat([majority_class,minority_upsampled])
    X_train_upsampled = upsampled.drop(columns='stroke')
    y_train_upsamples = upsampled['stroke']
    return X_train_upsampled , y_train_upsamples,X_test,y_test

def model(x_train,y_train,x_test,y_test):
    models = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),SGDClassifier(),SVC()]
    gridSearchModels = models.copy()
    best = {}
    for i in models:
        model =i.fit(x_train,y_train)
        score = i.score(x_test,y_test)
        report=classification_report(y_test, i.predict(x_test))
        if not best  or score > best.get("score",0):
            best["model"] = model
            best["score"]= score
        print(f'{model} score:{score}\n{report}\n')
    print(f'Saving the best model')
    print(f"Best {best['model']} -- Score {best['score']}")
    path_save = os.environ.get("PATH_SAVE")
    joblib.dump(best['model'], f"{path_save}/model.pkl")

def main():
    file_path = os.environ.get('PATH_CSV')
    df = extract_data(file_path)
    clean_df = transform_data(df)
    feature,target = split_features_target(clean_df)
    x_train,y_train,x_test,y_test=split_train_test(feature,target)
    model(x_train,y_train,x_test,y_test)


main()

