import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("📊 Data Science App with GitHub Actions")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/iris.csv")

df = load_data()
st.write("### Dataset Preview")
st.dataframe(df.head())

# Show basic stats
st.write("### Dataset Description")
st.write(df.describe())

# Visualize data
st.write("### Pairplot of Features")
fig, ax = plt.subplots(figsize=(6, 4))
sns.pairplot(df, hue="species")
st.pyplot(fig)
