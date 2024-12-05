# Import all libraries
import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('model_SVM.pkl', 'rb'))

# Function for prediction
def crop_prediction(input_data):
    # Convert input data into a NumPy array
    input_data_as_nparray = np.asarray(input_data, dtype=float)
    # Reshape the data since there is only one instance
    input_data_reshaped = input_data_as_nparray.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

def main():
    # Custom CSS styles for advanced visuals
    st.markdown("""
    <style>
    body {
        background-color: #f9fbfd;
        font-family: Arial, sans-serif;
    }
    .main-title {
        background-color: #023e8a;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 30px;
        color: white;
        font-weight: bold;
        margin-bottom: 30px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .input-section {
        padding: 20px;
        background-color: #dfeffc;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #c8f7dc;
        border: 2px solid #2d6a4f;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #1b4332;
    }
    .footer {
        text-align: center;
        color: #023e8a;
        font-size: 14px;
        margin-top: 50px;
    }
    .icon {
        height: 100px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # App title
    st.markdown('<div class="main-title">🌾 Advanced Crop Recommendation Assistant 🌱</div>', unsafe_allow_html=True)

    # Display header icon
 

    # Input fields in columns
    st.markdown('<div class="section-header">Enter Soil and Weather Details:</div>', unsafe_allow_html=True)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            Nitrogen = st.text_input('Nitrogen Content (mg/kg):', placeholder="e.g., 90")
            Phosphorus = st.text_input('Phosphorus Level (mg/kg):', placeholder="e.g., 42")
            potassium = st.text_input('Potassium Value (mg/kg):', placeholder="e.g., 43")
        with col2:
            Temperature = st.text_input('Temperature (°C):', placeholder="e.g., 22.5")
            Humidity = st.text_input('Humidity (%):', placeholder="e.g., 80")
            Ph = st.text_input('Soil pH:', placeholder="e.g., 6.5")
        Rainfall = st.text_input('Rainfall (mm):', placeholder="e.g., 120")

    # Interactive prediction button
    if st.button('🌟 Recommend a Crop'):
        try:
            # Perform prediction
            prediction = crop_prediction([Nitrogen, Phosphorus, potassium, Temperature, Humidity, Ph, Rainfall])
            # Display the result in a styled box
            st.markdown(f'<div class="prediction-box">🌟 Recommended Crop: <b>{prediction}</b></div>', unsafe_allow_html=True)
            # Show a relevant icon
            
        except Exception as e:
            st.error("Error in prediction. Ensure all inputs are valid.")

    # Footer
    st.markdown("""
    <hr>
    <div class="footer">
        
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
