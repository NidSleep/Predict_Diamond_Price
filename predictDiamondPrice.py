import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load your trained scikit-learn KNeighborsRegressor model here
model = joblib.load('model2.pkl')

# Dictionary to encode color and clarity
color_encoding = {"b'J'": 1, "b'I'": 2, "b'H'": 3, "b'G'": 4, "b'F'": 5, "b'E'": 6, "b'D'": 7}
clarity_encoding = {"b'I1'": 1, "b'SI2'": 2, "b'SI1'": 3, "b'VS2'": 4, "b'VS1'": 5, "b'VVS2'": 6, "b'VVS1'": 7, "b'IF'": 8}

def encode_color(color):
    return color_encoding.get(color, 0)  # Default to 0 if color is not found

def encode_clarity(clarity):
    return clarity_encoding.get(clarity, 0)  # Default to 0 if clarity is not found

def predict_price(carat, cut, color, clarity, depth, table, x, y, z):
    try:
        color = encode_color(color)  # Encode color
        clarity = encode_clarity(clarity)  # Encode clarity

        # Make price predictions using the model
        input_data = np.array([carat, color, clarity, x, y, z]).reshape(1, -1)
        predicted_price = model.predict(input_data)

        return predicted_price[0]
    except Exception as e:
        return None

st.title("Diamond Price Prediction")

# Input fields
carat = st.slider("Carat:", 0.2, 5.0, 0.2, 0.1)
cut = st.selectbox("Cut:", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color:", ["J", "I", "H", "G", "F", "E", "D"])
clarity = st.selectbox("Clarity:", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
depth = st.slider("Depth:", 40.0, 80.0, 40.0, 0.1)
table = st.slider("Table:", 50.0, 80.0, 50.0, 0.1)
x = st.slider("Length (mm):", 0.0, 10.0, 0.0, 0.1)
y = st.slider("Width (mm):", 0.0, 10.0, 0.0, 0.1)
z = st.slider("Depth (mm):", 0.0, 10.0, 0.0, 0.1)

if st.button("Predict Diamond Price"):
    predicted_price = predict_price(carat, cut, color, clarity, depth, table, x, y, z)
    if predicted_price is not None:
        st.success(f"Predicted Price: ${predicted_price:.2f}")
    else:
        st.error("An error occurred while making predictions.")

# Upload a dataset for bulk prediction
st.header("Bulk Import and Predict")
uploaded_file = st.file_uploader("Upload a CSV or Excel file:", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # Predict prices for each row and add a new column 'Predicted Price' to the dataset
        df['Predicted Price'] = df.apply(lambda row: predict_price(row['carat'], row['cut'], row['color'], row['clarity'], row['depth'], row['table'], row['x'], row['y'], row['z']), axis=1)

        # Display the dataset with predictions
        st.write(df)
    except Exception as e:
        st.error(f"An error occurred while importing the dataset: {str(e)}")
