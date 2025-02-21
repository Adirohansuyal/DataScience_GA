import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Sidebar Navigation
st.sidebar.title("🌟 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Train Model", "Predict"])

# Home Page
if page == "Home":
    st.title("🌺 Iris Flower Classification")
    st.write("This app classifies **Iris flowers** using a **Random Forest model**.")
    st.image("https://source.unsplash.com/800x400/?flower", use_column_width=True)

# Data Analysis
elif page == "Data Analysis":
    st.title("📊 Data Analysis")
    st.write("### Sample Data")
    st.dataframe(df.head())

    # Feature Correlation Heatmap
    st.write("### Feature Correlation")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Pairplot of Features
    st.write("### Feature Distribution")
    fig = sns.pairplot(df, hue="species", palette="coolwarm")
    st.pyplot(fig)

# Model Training
elif page == "Train Model":
    st.title("⚙️ Train the Model")

    # Feature Selection
    X = df.drop(columns=['species'])
    y = df['species']

    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Display Accuracy
    st.success(f"✅ **Model Accuracy: {accuracy:.2f}**")

# Prediction Section
elif page == "Predict":
    st.title("🔮 Make a Prediction")

    st.sidebar.header("🌟 Predict New Flower")
    sepal_length = st.sidebar.slider("Sepal Length", float(df["sepal_length"].min()), float(df["sepal_length"].max()))
    sepal_width = st.sidebar.slider("Sepal Width", float(df["sepal_width"].min()), float(df["sepal_width"].max()))
    petal_length = st.sidebar.slider("Petal Length", float(df["petal_length"].min()), float(df["petal_length"].max()))
    petal_width = st.sidebar.slider("Petal Width", float(df["petal_width"].min()), float(df["petal_width"].max()))

    # Prediction Button
    if st.sidebar.button("Predict"):
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_data)
        st.sidebar.success(f"🌼 **Predicted Species: {prediction[0]}**")
