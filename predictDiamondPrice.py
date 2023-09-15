import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load your trained scikit-learn KNeighborsRegressor model here
# Replace 'your_model.pkl' with the actual model file path
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
        user_input = np.array([carat, x, y, z]).reshape(1, -1)
        predicted_price = model.predict(user_input)
        st.subheader(f'Predicted Price: ${predicted_price[0]:.2f}')
    except ValueError:
        st.error("Please enter valid numerical values for all fields.")

# Bulk Import button
if st.button("Bulk Import and Predict"):
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                df = pd.read_excel(uploaded_file, header=None)
            else:
                df = pd.read_csv(uploaded_file, header=None)

            # Predict prices for each row and add a new column 'Predicted Price' to the dataset
            df['Predicted Price'] = df.apply(predict_price, axis=1)

            # Display the dataset with predictions
            st.subheader("Imported Dataset with Predictions")
            st.write(df)

            # Save the dataset with predictions to a new CSV file
            df.to_csv('predicted_dataset.csv', index=False)

            # Provide a download link for the user
            st.markdown("### Download the Predicted Dataset")
            st.markdown("[Download CSV](predicted_dataset.csv)")

        except Exception as e:
            st.error(f"An error occurred while importing the dataset: {str(e)}")
    
