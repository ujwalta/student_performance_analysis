🎓 Student Performance Analysis & Prediction App

An interactive web app to explore student performance data and predict math scores using Machine Learning (Random Forest), built with Python & Streamlit.

🔹 Features

Dataset Overview: View raw data and summary statistics

Exploratory Data Analysis (EDA):

Average score distribution

Test preparation vs performance

Lunch type vs performance

ML Prediction:

Predict Math scores based on Reading, Writing, and background info

Random Forest Regressor for high accuracy

Feature importance visualization

Interactive Streamlit UI: Input scores and get predictions in real time

🔹 Tech Stack

Python 3.x

Streamlit

Pandas & NumPy

Matplotlib & Seaborn

Scikit-learn (Random Forest Regressor)

🔹 Dataset

Source: StudentsPerformance.csv

Features:

Gender

Race/Ethnicity

Parental Education

Lunch Type

Test Preparation

Math, Reading, Writing Scores

🔹 How to Run Locally

Clone the repository:

git clone https://github.com/ujwalta/students_performance_analysis.git


Navigate to the project folder:

cd students_performance_analysis


Install dependencies:

pip install -r requirements.txt


Run Streamlit app:

streamlit run app.py


Open your browser at the URL shown in terminal (usually http://localhost:8501)

🔹 ML Model

Algorithm: Random Forest Regressor

Evaluation Metrics:

MAE (Mean Absolute Error)

R² Score

Features used for prediction: Reading, Writing, Gender, Lunch, Test Preparation, Parental Education, Race/Ethnicity



🔹 Future Enhancements

Add cross-validation

Deploy on Streamlit Cloud

Include confidence intervals for predictions

Add correlation heatmaps & more interactive visualizations

🔹 Author

Ujwalta Khanal

GitHub: https://github.com/ujwalta

LinkedIn: (add your LinkedIn URL)
