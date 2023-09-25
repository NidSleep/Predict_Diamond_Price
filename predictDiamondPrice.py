import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64

from scipy.stats import norm

# Load your trained scikit-learn KNeighborsRegressor model here
model = joblib.load('model.pkl')

# Dictionary to encode color and clarity
color_encoding = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
clarity_encoding = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}
# Calculate mean squared error (MSE) as a measure of prediction variance
mse = 302566.6335987021

# Calculate the margin of error (standard error)
std_error = np.sqrt(mse)

def encode_color(color):
    return color_encoding.get(color, 0)  # Default to 0 if color is not found

def encode_clarity(clarity):
    return clarity_encoding.get(clarity, 0)  # Default to 0 if clarity is not found

def predict_price(carat, color_encoded, clarity_encoded, depth, table, x, y, z):
    try:
        # Make price predictions using the model
        input_data = np.array([carat, color_encoded, clarity_encoded, x, y, z]).reshape(1, -1)
        predicted_price = model.predict(input_data)
        return predicted_price[0]
    except Exception as e:
        return None

st.title("Diamond Price Prediction")

# Input fields
carat = st.slider("Carat:", 0.2, 3.0, 0.2, 0.01)
cut = st.selectbox("Cut:", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color:", ["J", "I", "H", "G", "F", "E", "D"])
clarity = st.selectbox("Clarity:", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
depth = st.slider("Depth:", 40.0, 80.0, 40.0, 0.01)
table = st.slider("Table:", 50.0, 80.0, 50.0, 0.01)
x = st.slider("Length (mm):", 0.0, 10.0, 0.0, 0.01)
y = st.slider("Width (mm):", 0.0, 10.0, 0.0, 0.01)
z = st.slider("Depth (mm):", 0.0, 10.0, 0.0, 0.01)

if st.button("Predict Diamond Price"):
    predicted_price = predict_price(carat, encode_color(color), encode_clarity(clarity), depth, table, x, y, z)
    # Define the desired confidence level (e.g., 95%)
    confidence_level = 0.95

    # Calculate the critical value (z-score) for the desired confidence level
    critical_value = norm.ppf((1 + confidence_level) / 2)  # For a two-tailed test
    
    # Calculate the margin of error
    margin_of_error = std_error * critical_value
    if predicted_price is not None:
        # Calculate lower and upper bounds of the prediction interval
        lower_bound = predicted_price - margin_of_error
        upper_bound = predicted_price + margin_of_error
        st.success(f"Predicted Price: ${predicted_price:.2f}")
        #st.write(f" ± ${margin_of_error:.2f}")

    else:
        st.error("An error occurred while making predictions.")

# Pre-processing of the data in columns
color_encoding = {"b'J'": 1, "b'I'": 2, "b'H'": 3, "b'G'": 4, "b'F'": 5, "b'E'": 6, "b'D'": 7}
clarity_encoding = {"b'I1'": 1, "b'SI2'": 2, "b'SI1'": 3, "b'VS2'": 4, "b'VS1'": 5, "b'VVS2'": 6, "b'VVS1'": 7, "b'IF'": 8}

def encode_color_bulk(color):
    return color_encoding.get(color, 0)  # Default to 0 if color is not found

def encode_clarity_bulk(clarity):
    return clarity_encoding.get(clarity, 0)  # Default to 0 if clarity is not found

st.header("Bulk Import and Predict")
uploaded_file = st.file_uploader("Upload a CSV or Excel file:", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # Encode 'color' and 'clarity' columns
        df['color_encoded'] = df['color'].apply(encode_color_bulk)
        df['clarity_encoded'] = df['clarity'].apply(encode_clarity_bulk)

        # Predict prices for each row and add a new column 'Predicted Price' to the dataset
        df['Predicted Price'] = df.apply(lambda row: predict_price(row['carat'], row['color_encoded'], row['clarity_encoded'], row['depth'], row['table'], row['x'], row['y'], row['z']), axis=1)

        # Display the dataset with predictions
        st.write(df[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z', 'Predicted Price']])

        # Create a function to download the CSV file
        def download_csv(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predicted_prices.csv">Download CSV</a>'
            return href

        # Add a download button for the CSV file
        st.markdown(download_csv(df), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred while importing the dataset: {str(e)}")
