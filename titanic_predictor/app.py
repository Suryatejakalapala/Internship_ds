import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
import seaborn as sns
import os

st.set_page_config(page_title="Exploring the Titanic Dataset", page_icon=":ship:")

intro = """
## Welcome to my Streamlit website that explores the Titanic dataset!

The Titanic dataset contains information about the passengers who were aboard the Titanic when it sank on April 15, 1912. The dataset includes various attributes such as the passenger's name, age, gender, ticket class, fare and whether or not they survived the disaster.

In this website, we will explore the Titanic dataset and gain insights into the passengers' demographics and survival rates. We will use various data visualization techniques and machine learning algorithms to analyze the data and make predictions.

Let's dive in!
"""
FILE_PATH = pathlib.Path(__file__)
FILE_DIR = FILE_PATH.cwd()
dir_of_interest = FILE_DIR / "resources"

IMAGE_PATH = dir_of_interest / "images" / "titanic.jpg"
DATA_PATH = dir_of_interest / "data" / "titanic.csv"

img = image.imread(IMAGE_PATH)
st.image(img)

titanic_data = pd.read_csv(DATA_PATH)
st.markdown(intro)

# Display the first 10 rows of the dataset
st.subheader("Titanic Dataset")
st.dataframe(titanic_data.head(10))

# Display a bar chart of the passenger class distribution
st.subheader("Passenger Class Distribution")
class_dist = titanic_data['Pclass'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=class_dist.index, y=class_dist.values, ax=ax)
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Count")
st.pyplot(fig)

# Display a pie chart of the passenger gender distribution
st.subheader("Passenger Gender Distribution")
gender_dist = titanic_data['Sex'].value_counts()
fig, ax = plt.subplots()
ax.pie(gender_dist.values, labels=gender_dist.index, autopct='%1.1f%%')
ax.set_title("Passenger Gender Distribution")
st.pyplot(fig)

