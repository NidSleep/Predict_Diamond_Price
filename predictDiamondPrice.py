import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load your trained scikit-learn KNeighborsRegressor model here
model = joblib.load('model.pkl')

st.title("Diamond Price Prediction")

# Use st.slider for carat input
carat = st.slider("Carat:", min_value=0.2, max_value=5.0, step=0.01, value=0.2)

# Use st.slider for depth input
depth = st.slider("Depth:", min_value=40.0, max_value=80.0, step=0.01, value=40.0)

cut = st.selectbox("Cut:", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color:", ["J", "I", "H", "G", "F", "E", "D"])
clarity = st.selectbox("Clarity:", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

# Use st.slider for table input
table = st.slider("Table:", min_value=50.0, max_value=80.0, step=0.01, value=50.0)

# Use st.slider for length (mm) input
x = st.slider("Length (mm):", min_value=0.0, max_value=10.0, step=0.01, value=0.0)

# Use st.slider for width (mm) input
y = st.slider("Width (mm):", min_value=0.0, max_value=10.0, step=0.01, value=0.0)

# Use st.slider for depth (mm) input
z = st.slider("Depth (mm):", min_value=0.0, max_value=10.0, step=0.01, value=0.0)

# Predict button
if st.button("Predict Diamond Price"):
    try:
        # Create a DataFrame from the user input
        user_input = pd.DataFrame({
            'carat': [carat],
            'cut': [cut],
            'color': [color],
            'clarity': [clarity],
            'depth': [depth],
            'table': [table],
            'x': [x],
            'y': [y],
            'z': [z]
        })

        # Map categorical values to numerical codes (you can use a more sophisticated mapping if needed)
        user_input['cut'] = user_input['cut'].map({'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4})
        user_input['color'] = user_input['color'].map({'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6})
        user_input['clarity'] = user_input['clarity'].map({'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7})

        predicted_price = model.predict(user_input)
        st.subheader(f'Predicted Price: ${predicted_price[0]:.2f}')
    except ValueError:
        st.error("Please enter valid numerical values for all fields.")
