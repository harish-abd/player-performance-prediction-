import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and encoded data
with open("C:/Users/vkabd/Downloads/model_and_data.pkl", "rb") as file:
    data = pickle.load(file)

model = data["model"]
x = data["x"]

st.title("🏏 Player Run Class Predictor")

# Upload CSV file for actual batter and team names
uploaded_file = st.file_uploader("C:/Users/vkabd/Downloads/deliveries.csv/deliveries.csv.zip", type="csv")

if uploaded_file:
    # Read the uploaded CSV
    original_data = pd.read_csv(uploaded_file, on_bad_lines='skip')

    # Extract unique batter and bowling team names
    batters = sorted(original_data["batter"].dropna().unique())
    bowling_teams = sorted(original_data["bowling_team"].dropna().unique())

    # Create encoders (must match training order!)
    batter_encoder = {name: i for i, name in enumerate(batters)}
    team_encoder = {name: i for i, name in enumerate(bowling_teams)}

    # Streamlit dropdowns with actual names
    selected_batter = st.selectbox("Select Batter", batters)
    selected_team = st.selectbox("Select Bowling Team", bowling_teams)

    if st.button("Predict Run Class"):
        # Encode using created mappings
        if selected_batter in batter_encoder and selected_team in team_encoder:
            batter_encoded = batter_encoder[selected_batter]
            team_encoded = team_encoder[selected_team]

            input_data = np.array([[batter_encoded, team_encoded]])
            prediction = model.predict(input_data)[0]

            class_map = {
                0: "Low (0–19 runs)",
                1: "Medium (20–49 runs)",
                2: "High (50+ runs)"
            }

            st.success(
                f"🧢 **{selected_batter}** vs 🏏 **{selected_team}** ➤ Predicted Run Class: **{class_map[prediction]}**"
            )
        else:
            st.error("Selected batter or team not found in training data.")
else:
    st.info("📂 Please upload the `deliveries.csv` file to continue.")
