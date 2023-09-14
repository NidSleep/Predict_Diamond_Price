import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib

# Load your trained scikit-learn KNeighborsRegressor model here
# Replace 'your_model.pkl' with the actual model file path
import joblib

# Train the XGBRegressor
xr = XGBRegressor()
xr.fit(x_train, y_train)

# Save the trained model to a file
joblib.dump(xr, 'model.pkl')
model = joblib.load('model.pkl')

def predict_diamond_price():
    try:
        carat = carat_scale.get()  # Use the scale value
        cut = cut_combobox.get()
        color = color_combobox.get()
        clarity = clarity_combobox.get()
        depth = depth_scale.get()  # Use the scale value
        table = table_scale.get()  # Use the scale value
        x = x_scale.get()  # Use the scale value
        y = y_scale.get()  # Use the scale value
        z = z_scale.get()  # Use the scale value

        # Encode categorical variables (replace this with your actual encoding)
        cut_encoding = 0  # Replace with actual encoding logic
        color_encoding = 0  # Replace with actual encoding logic
        clarity_encoding = 0  # Replace with actual encoding logic

        user_input = np.array([carat, x, y, z]).reshape(1, -1)

        predicted_price = model.predict(user_input)

        # Update the label with a larger font and red color
        predicted_label.config(text=f'Predicted Price: ${predicted_price[0]:.2f}', font=("Helvetica", 16), fg="red")

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values for all fields.")

# Create the Tkinter window
window = tk.Tk()
window.title("Diamond Price Prediction")
window.geometry("400x570")  # Set the initial window size

# Labels and input fields
carat_label = tk.Label(window, text="Carat:")
carat_label.pack()

# Use ttk.Scale for carat input
carat_scale = ttk.Scale(window, from_=0.2, to=5.01, length=200, orient="horizontal")
carat_scale.pack()

# Label to display carat value
carat_value_label = ttk.Label(window, text="0.2")
carat_value_label.pack()

def update_carat_value(value):
    carat_value_label.config(text=f"{value:.2f}")

carat_scale.config(command=lambda value: update_carat_value(carat_scale.get()))

# Depth
depth_label = tk.Label(window, text="Depth:")
depth_label.pack()
depth_scale = ttk.Scale(window, from_=40, to=80, length=200, orient="horizontal")
depth_scale.pack()
depth_value_label = ttk.Label(window, text="40")
depth_value_label.pack()

cut_label = tk.Label(window, text="Cut:")
cut_label.pack()
cut_options = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
cut_combobox = ttk.Combobox(window, values=cut_options)
cut_combobox.pack()

color_label = tk.Label(window, text="Color:")
color_label.pack()
color_options = ["J", "I", "H", "G", "F", "E", "D"]
color_combobox = ttk.Combobox(window, values=color_options)
color_combobox.pack()

clarity_label = tk.Label(window, text="Clarity:")
clarity_label.pack()
clarity_options = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
clarity_combobox = ttk.Combobox(window, values=clarity_options)
clarity_combobox.pack()

def update_depth_value(value):
    depth_value_label.config(text=f"{value:.2f}")

depth_scale.config(command=lambda value: update_depth_value(depth_scale.get()))

# Table
table_label = tk.Label(window, text="Table:")
table_label.pack()
table_scale = ttk.Scale(window, from_=50, to=80, length=200, orient="horizontal")
table_scale.pack()
table_value_label = ttk.Label(window, text="50")
table_value_label.pack()

def update_table_value(value):
    table_value_label.config(text=f"{value:.2f}")

table_scale.config(command=lambda value: update_table_value(table_scale.get()))

# Length (mm)
x_label = tk.Label(window, text="Length (mm):")
x_label.pack()
x_scale = ttk.Scale(window, from_=0, to=10, length=200, orient="horizontal")
x_scale.pack()
x_value_label = ttk.Label(window, text="0")
x_value_label.pack()

def update_x_value(value):
    x_value_label.config(text=f"{value:.2f}")

x_scale.config(command=lambda value: update_x_value(x_scale.get()))

# Width (mm)
y_label = tk.Label(window, text="Width (mm):")
y_label.pack()
y_scale = ttk.Scale(window, from_=0, to=10, length=200, orient="horizontal")
y_scale.pack()
y_value_label = ttk.Label(window, text="0")
y_value_label.pack()

def update_y_value(value):
    y_value_label.config(text=f"{value:.2f}")

y_scale.config(command=lambda value: update_y_value(y_scale.get()))

# Depth (mm)
z_label = tk.Label(window, text="Depth (mm):")
z_label.pack()
z_scale = ttk.Scale(window, from_=0, to=10, length=200, orient="horizontal")
z_scale.pack()
z_value_label = ttk.Label(window, text="0")
z_value_label.pack()

def update_z_value(value):
    z_value_label.config(text=f"{value:.2f}")

z_scale.config(command=lambda value: update_z_value(z_scale.get()))

# Rest of the input fields and scales (similar to carat)

# Predict button
predict_button = tk.Button(window, text="Predict Diamond Price", command=predict_diamond_price)
predict_button.pack()

# Display predicted price
predicted_label = tk.Label(window, text="", font=("Helvetica", 16), fg="red")
predicted_label.pack()

# Start the Tkinter main loop
window.mainloop()