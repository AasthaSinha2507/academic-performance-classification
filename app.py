import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Academic Performance Classification", layout="wide")
st.title("🎓 Academic Performance Classification System")
st.write("Predict whether a student will Pass or Fail based on academic features.")

# Sidebar for inputs
st.sidebar.header("Enter Student Details")

gender = st.sidebar.selectbox("Gender", ["female", "male"])
race = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.sidebar.selectbox("Parental Level of Education",
                                  ["some high school", "high school", "some college",
                                   "associate's degree", "bachelor's degree", "master's degree"])
lunch = st.sidebar.selectbox("Lunch", ["standard", "free/reduced"])
test_prep = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])

math_score = st.sidebar.slider("Math Score", 0, 100, 50)
reading_score = st.sidebar.slider("Reading Score", 0, 100, 50)
writing_score = st.sidebar.slider("Writing Score", 0, 100, 50)

# Encode categorical variables
gender_encoded = 1 if gender == "male" else 0
lunch_encoded = 1 if lunch == "standard" else 0
test_prep_encoded = 1 if test_prep == "completed" else 0

race_dict = {"group A":0,"group B":1,"group C":2,"group D":3,"group E":4}
edu_dict = {"some high school":5,"high school":3,"some college":4,
            "associate's degree":0,"bachelor's degree":1,"master's degree":2}

race_encoded = race_dict[race]
parent_edu_encoded = edu_dict[parent_edu]

# Create input dataframe for prediction
input_data = pd.DataFrame([[gender_encoded, race_encoded, parent_edu_encoded, lunch_encoded, test_prep_encoded,
                            math_score, reading_score, writing_score]],
                          columns=["gender", "race/ethnicity", "parental level of education",
                                   "lunch", "test preparation course",
                                   "math score", "reading score", "writing score"])

# Prediction
if st.button("Predict Result"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ Student is likely to PASS")
    else:
        st.error("❌ Student is likely to FAIL")

# Dynamic Graphs based on inputs
st.header("📊 Dynamic Visualization Based on Your Inputs")

# Average score calculation
average_score = (math_score + reading_score + writing_score)/3

# Create a small dataframe for dynamic plotting
dynamic_df = pd.DataFrame({
    "Subjects": ["Math", "Reading", "Writing", "Average"],
    "Score": [math_score, reading_score, writing_score, average_score]
})

# Plot
fig, ax = plt.subplots()
sns.barplot(x="Subjects", y="Score", data=dynamic_df, palette="viridis", ax=ax)
ax.set_ylim(0, 100)
ax.set_title(f"Scores for This Student ({gender.title()})")
st.pyplot(fig)

# Optional: Show raw input values
st.subheader("Student Input Details")
st.write(dynamic_df)