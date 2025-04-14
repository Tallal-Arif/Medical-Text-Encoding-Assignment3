
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import ast

st.set_page_config(page_title="Disease KNN Classifier", layout="wide")

st.title("🧬 Disease Classification using KNN")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("data/disease_features.csv")
    df['Risk Factors'] = df['Risk Factors'].apply(lambda x: " ".join(ast.literal_eval(x)) if isinstance(x, str) else "")
    df['Symptoms'] = df['Symptoms'].apply(lambda x: " ".join(ast.literal_eval(x)) if isinstance(x, str) else "")
    df['Signs'] = df['Signs'].apply(lambda x: " ".join(ast.literal_eval(x)) if isinstance(x, str) else "")
    df['Subtypes'] = df['Subtypes'].apply(lambda x: " ".join(ast.literal_eval(x).keys()) if isinstance(x, str) else "")
    return df

df = load_data()

# TF-IDF transform
vec_rf = TfidfVectorizer().fit(df['Risk Factors'])
vec_symptoms = TfidfVectorizer().fit(df['Symptoms'])
vec_signs = TfidfVectorizer().fit(df['Signs'])
vec_subtypes = TfidfVectorizer().fit(df['Subtypes'])

X = hstack([
    vec_rf.transform(df['Risk Factors']),
    vec_symptoms.transform(df['Symptoms']),
    vec_signs.transform(df['Signs']),
    vec_subtypes.transform(df['Subtypes'])
])
y = df['Disease']

# Sidebar
st.sidebar.header("Model Settings")
k = st.sidebar.slider("k (number of neighbors)", min_value=1, max_value=10, value=3)
metric = st.sidebar.selectbox("Distance metric", ["euclidean", "manhattan", "cosine"])

# User Input
st.subheader("🔍 Input Disease Details")

risk_input = st.text_input("Risk Factors", "hypertension diabetes")
symptoms_input = st.text_input("Symptoms", "chest pain fatigue")
signs_input = st.text_input("Signs", "")
subtypes_input = st.text_input("Subtypes", "STEMI")

if st.button("Predict Disease"):
    x_input = hstack([
        vec_rf.transform([risk_input]),
        vec_symptoms.transform([symptoms_input]),
        vec_signs.transform([signs_input]),
        vec_subtypes.transform([subtypes_input])
    ])
    model = make_pipeline(StandardScaler(with_mean=False), KNeighborsClassifier(n_neighbors=k, metric=metric))
    model.fit(X, y)
    prediction = model.predict(x_input)
    st.success(f"🩺 Predicted Disease: **{prediction[0]}**")
