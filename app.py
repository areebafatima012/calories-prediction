import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Calories Predictor", page_icon="🔥", layout="wide")

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
h1 {
    text-align: center;
    color: #ff4b4b;
}
.stButton>button {
    width: 100%;
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    border-radius: 10px;
}
.result-box {
    background-color: #022c22;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    color: #4ade80;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #
st.markdown("<h1>🔥 Calories Burn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>AI Powered Fitness Tracker</p>", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("🏃 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Prediction"])

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("calories.csv")

df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
df.drop(['User_ID'], axis=1, inplace=True)

# ---------------- MODEL ---------------- #
X = df.drop(['Calories'], axis=1)
y = df['Calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBRegressor()
model.fit(X_train, y_train)

# ================= DASHBOARD ================= #
if page == "Dashboard":

    st.subheader("📊 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Data Preview")
        st.dataframe(df.head())

    with col2:
        st.write("### Statistics")
        st.write(df.describe())

    # ----------- GRAPHS ----------- #
    st.subheader("📈 Data Visualizations")

    col3, col4 = st.columns(2)

    with col3:
        st.write("#### 🔥 Calories Distribution")
        fig1, ax1 = plt.subplots()
        sb.histplot(df['Calories'], kde=True)
        st.pyplot(fig1)

    with col4:
        st.write("#### ⏱ Duration vs Calories")
        fig2, ax2 = plt.subplots()
        sb.scatterplot(x=df['Duration'], y=df['Calories'])
        st.pyplot(fig2)

    col5, col6 = st.columns(2)

    with col5:
        st.write("#### ❤️ Heart Rate vs Calories")
        fig3, ax3 = plt.subplots()
        sb.scatterplot(x=df['Heart_Rate'], y=df['Calories'])
        st.pyplot(fig3)

    with col6:
        st.write("#### 🌡 Body Temp vs Calories")
        fig4, ax4 = plt.subplots()
        sb.scatterplot(x=df['Body_Temp'], y=df['Calories'])
        st.pyplot(fig4)

    # Heatmap
    st.subheader("🔥 Correlation Heatmap")
    fig5, ax5 = plt.subplots()
    sb.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig5)

# ================= PREDICTION ================= #
elif page == "Prediction":

    st.subheader("🧍 Enter Your Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["male", "female"])
        age = st.number_input("Age", 1, 100)
        height = st.number_input("Height (cm)")

    with col2:
        weight = st.number_input("Weight (kg)")
        duration = st.number_input("Duration (min)")
        heart_rate = st.number_input("Heart Rate")

    body_temp = st.number_input("Body Temperature")

    gender = 1 if gender == "female" else 0

    if st.button("🔥 Predict Calories Burned"):

        input_data = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)

        st.markdown("---")

        st.markdown(
            f"<div class='result-box'>🔥 Estimated Calories Burned: <br><b>{prediction[0]:.2f}</b></div>",
            unsafe_allow_html=True
        )

        st.balloons()

        st.markdown(
            "<p style='text-align:center; color:#4ade80;'>💪 Stay Fit | 🏃 Keep Moving | ❤️ Stay Healthy</p>",
            unsafe_allow_html=True
        )