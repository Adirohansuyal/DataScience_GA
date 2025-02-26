import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Custom CSS for Flowery Background
import streamlit as st

# Custom CSS for Gradient Royal Blue Background
page_bg = """
<style>

    body {
        background: url('https://source.unsplash.com/1600x900/?stars,night-sky,galaxy');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(0, 0, 0, 0.7); /* Dark overlay for readability */
        border-radius: 10px;
        padding: 20px;
        color: white;
    }
    
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)


# Load Dataset
@st.cache_data
def load_data():
    url = "data/iris.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Train Model Function
@st.cache_resource
def train_model(model_type="Random Forest", n_estimators=100, kernel="rbf"):
    X = df.drop(columns=['species'])
    y = df['species']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=n_estimators)
    else:
        model = SVC(kernel=kernel, probability=True)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

# App Layout
st.title("🌺 Iris Flower Classification Model")
st.write("A Streamlit app demonstrating GitHub Actions with automated feature updates (CI/CD).")

# Sidebar Navigation
st.sidebar.header("🔍 Select Options below")
menu = st.sidebar.radio("Go to", ["📊 Data Overview", "🔬 Model Training", "🧠 Prediction"])

# 1️⃣ Data Overview
if menu == "📊 Data Overview":
    st.subheader("🔍 Data Overview")
    st.dataframe(df.head())
    
    # Data Visualization
    st.subheader("📊 Feature Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    df.hist(ax=ax, bins=15)
    st.pyplot(fig)
    
    st.subheader("📌 Feature Relationships")
    pairplot_fig = sns.pairplot(df, hue="species")
    st.pyplot(pairplot_fig)

# 2️⃣ Model Training
elif menu == "🔬 Model Training":
    st.subheader("🔬 Train ML Model")
    
    # Select Model Type
    model_type = st.selectbox("Select Model", ["Random Forest", "SVM"])
    
    if model_type == "Random Forest":
        n_estimators = st.slider("Number of Estimators", 50, 300, step=50, value=100)
        model, accuracy = train_model(model_type, n_estimators=n_estimators)
    else:
        kernel = st.selectbox("Kernel Type", ["linear", "poly", "rbf", "sigmoid"])
        model, accuracy = train_model(model_type, kernel=kernel)
    
    st.write(f"### Model Accuracy: **{accuracy:.2f}**")

# 3️⃣ Prediction
elif menu == "🧠 Prediction":
    st.subheader("🌟 Predict New Flower 🌸")

    # Train a default model if not already trained
    model, _ = train_model()  # Default: Random Forest with 100 estimators

    # Input Fields
    sepal_length = st.slider("Sepal Length", float(df["sepal_length"].min()), float(df["sepal_length"].max()))
    sepal_width = st.slider("Sepal Width", float(df["sepal_width"].min()), float(df["sepal_width"].max()))
    petal_length = st.slider("Petal Length", float(df["petal_length"].min()), float(df["petal_length"].max()))
    petal_width = st.slider("Petal Width", float(df["petal_width"].min()), float(df["petal_width"].max()))

    # Prediction
    if st.button("Predict"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)

        st.success(f"Predicted Species: **{prediction[0]}**")
        st.write(f"Confidence: **{np.max(probabilities) * 100:.2f}%**")

# Footer Section
footer = """
<style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        font-weight: bold;
        color: black;
    }
    footer {visibility: hidden;} /* Hide default Streamlit footer */
</style>

<div class="footer">
    Designed by Aditya Suyal (Not a proper development software. Trial Only!) | 💡 
    <a href="https://github.com/Adirohansuyal/DataScience_GA" target="_blank">View on GitHub</a>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
