import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Student Performance App", layout="wide")
st.title("🎓 Student Performance Analysis & Prediction ")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("StudentsPerformance.csv")

df = load_data()

# Rename columns for simplicity
df.columns = [
    'gender', 'race', 'parent_education',
    'lunch', 'test_prep',
    'math', 'reading', 'writing'
]

# Feature engineering
df['average_score'] = df[['math', 'reading', 'writing']].mean(axis=1)

# ---------------- SIDEBAR ----------------
st.sidebar.header("📌 Navigation")
page = st.sidebar.selectbox(
    "Select Section",
    ["Dataset Overview", "EDA", "ML Prediction"]
)

# ---------------- DATASET OVERVIEW ----------------
if page == "Dataset Overview":
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(20))
    st.write("Shape:", df.shape)
    st.write("Summary Statistics:")
    st.write(df.describe())

# ---------------- EDA ----------------
elif page == "EDA":
    st.subheader("📈 Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Average Score Distribution")
        fig = plt.figure()
        sns.histplot(df['average_score'], kde=True)
        st.pyplot(fig)

    with col2:
        st.write("Test Preparation vs Average Score")
        fig = plt.figure()
        sns.boxplot(x='test_prep', y='average_score', data=df)
        st.pyplot(fig)

    st.write("Lunch Type vs Average Score")
    fig = plt.figure()
    sns.boxplot(x='lunch', y='average_score', data=df)
    st.pyplot(fig)

# ---------------- ML PREDICTION ----------------
else:
    st.subheader("🤖 Predict Math Score")

    st.write("Prediction using reading & writing scores + optional background features")

    # Encode categorical features
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Features: reading + writing + background info
    feature_cols = ['reading', 'writing'] + [col for col in df_encoded.columns 
                                             if col.startswith('gender_') 
                                             or col.startswith('lunch_') 
                                             or col.startswith('test_prep_') 
                                             or col.startswith('parent_education_') 
                                             or col.startswith('race_')]
    X = df_encoded[feature_cols]
    y = df['math']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown("### 📊 Model Performance")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # Feature Importance
    st.markdown("### 🔑 Feature Importance")
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(importance)

    st.markdown("---")
    st.markdown("### 🧑‍🎓 Enter Student Details")

    # User inputs
    reading = st.slider("Reading Score", 0, 100, 70)
    writing = st.slider("Writing Score", 0, 100, 70)

    gender = st.selectbox("Gender", df['gender'].unique())
    lunch = st.selectbox("Lunch Type", df['lunch'].unique())
    test_prep = st.selectbox("Test Preparation", df['test_prep'].unique())
    parent_edu = st.selectbox("Parental Education", df['parent_education'].unique())
    race = st.selectbox("Race/Ethnicity", df['race'].unique())

    # Prepare input
    input_df = pd.DataFrame([{
        'reading': reading,
        'writing': writing,
        'gender': gender,
        'lunch': lunch,
        'test_prep': test_prep,
        'parent_education': parent_edu,
        'race': race
    }])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    if st.button("Predict Math Score"):
        prediction = model.predict(input_encoded)[0]
        st.success(f"📐 Predicted Math Score: {prediction:.2f}")
