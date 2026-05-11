#write streamlit code to display the dataframe and the model prediction
import streamlit as st
import pandas as pd
import numpy as np
# Load the dataset
df = pd.read_csv('dataset.csv')
st.title("Telecom Churn Prediction")
st.subheader("Dataset")
st.dataframe(df.head())
# separating input and output variables
x = df.drop('Churn', axis=1)
y = df['Churn']
# converting categorical variables to numerical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])
# separating input and output variables
x = df.drop('Churn', axis=1)
y = df['Churn']
#splitting training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# training the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
st.success("Model trained successfully")
# making predictions
y_pred = model.predict(x_test)
st.write("Predicted values:\n",y_pred)
comparison = pd.DataFrame({"Actual":y_test.values.flatten(),"Predicted":y_pred})
st.subheader("Prediction Comparison")
st.dataframe(comparison)
# making a single prediction
st.subheader("Make a Single Prediction")
input_data = st.text_input("Enter the input data as comma separated values (excluding the target variable):")
if st.button("Predict"):
    input_data = np.array(input_data.split(",")).reshape(1,-1)
    probability = model.predict_proba(input_data)[0][1]
    st.write("Probability of churn:", probability)
    if probability > 0.6:
        st.write("Churn: Yes")
    else:
        st.write("Churn: No")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
st.subheader("Confusion Matrix")
st.write(cm)
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
accuracy = accuracy_score(y_test,y_pred)
st.write("Accuracy:",accuracy)
precision = precision_score(y_test,y_pred)
st.write("Precision:",precision)
recall = recall_score(y_test,y_pred)
st.write("Recall:",recall)
f1 = f1_score(y_test,y_pred)
st.write("F1 Score:",f1)
from sklearn.metrics import classification_report
report = classification_report(y_test,y_pred)
st.subheader("Classification Report")
st.text(report)