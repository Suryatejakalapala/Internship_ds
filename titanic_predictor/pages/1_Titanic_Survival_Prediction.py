import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
dir_of_interest = os.path.join(PARENT_DIR, "resources")

DATA_PATH = os.path.join(dir_of_interest, "data", "titanic.csv")

titanic_data = pd.read_csv(DATA_PATH)

# Preprocess the data
titanic_data = titanic_data.drop('Name', axis=1)
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].median())
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})

# Split the data into training and testing sets
X = titanic_data.drop(['Survived'], axis=1)
y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a machine learning model on the training data
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
model.fit(X_train, y_train)

# Create a Streamlit app
st.title('Titanic Survival Prediction')

# Add input fields for user to input data
pclass = st.selectbox('Pclass', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', min_value=0, max_value=100, value=30)
sibsp = st.selectbox('Siblings/Spouses Aboard', [0, 1, 2, 3, 4, 5, 6, 7, 8])
parch = st.selectbox('Parents/Children Aboard', [0, 1, 2, 3, 4, 5, 6])
fare = st.slider('Fare', min_value=0, max_value=500, value=50)

# Preprocess the user input
sex = 1 if sex == 'female' else 0
input_data = pd.DataFrame({'Pclass': [pclass], 'Sex': [sex], 'Age': [age], 'Siblings/Spouses Aboard': [sibsp], 'Parents/Children Aboard': [parch], 'Fare': [fare]})

# Make a prediction using the machine learning model
prediction = model.predict(input_data)[0]

# Display the prediction to the user
if prediction == 0:
    st.write('Unfortunately, you did not survive the Titanic disaster.')
else:
    st.write('Congratulations, you survived the Titanic disaster!')    
