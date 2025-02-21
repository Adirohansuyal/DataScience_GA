import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
@st.cache_data
def load_data():
    url = "data/iris.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Title & Dataset Preview
st.title("🌺 Iris Flower Classification by Aditya Suyal")
st.write("A simple ML model trained on the Iris dataset demonstrating Github Actions for future updates.")

st.write("### Sample Data")
st.dataframe(df.head())

# Feature Selection
X = df.drop(columns=['species'])
y = df['species']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"### Model Accuracy: **{accuracy:.2f}**")

# User Input for Prediction
st.sidebar.header("🌟 Predict New Flower 🌸 ")
sepal_length = st.sidebar.slider("Sepal Length", float(df["sepal_length"].min()), float(df["sepal_length"].max()))
sepal_width = st.sidebar.slider("Sepal Width", float(df["sepal_width"].min()), float(df["sepal_width"].max()))
petal_length = st.sidebar.slider("Petal Length", float(df["petal_length"].min()), float(df["petal_length"].max()))
petal_width = st.sidebar.slider("Petal Width", float(df["petal_width"].min()), float(df["petal_width"].max()))

# Prediction Button
if st.sidebar.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    st.sidebar.success(f"Predicted Species: **{prediction[0]}**")
