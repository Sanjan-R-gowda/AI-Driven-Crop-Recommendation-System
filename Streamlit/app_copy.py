# Import all libraries 
import numpy as np 
import pickle 
import pandas as pd
import streamlit as st 

# Load the dataset with farming methods
data = pd.read_excel('CropDataset.xlsx')  # Ensure the file is in the same directory or provide the correct path
  
# Function to fetch farming method
def get_farming_method(predicted_crop):
    farming_method = data[data['label'] == predicted_crop]['Farming Method'].values[0]
    return farming_method

# Main function
def main(): 
    # Title
    st.title('Crop Recommendation System') 
    
    # Dropdown for model selection
    options = ['Decision Tree', 'Random Forest', 'Support Vector Classifier', 'K-Nearest Neighbour', 'Ensemble']
    selected_value = st.selectbox("Select the model:", options)
    
    # Load the selected model
    if selected_value == 'Decision Tree':
        loaded_model = pickle.load(open('model_DT.pkl', 'rb')) 
    elif selected_value == 'Random Forest':
        loaded_model = pickle.load(open('model_RF.pkl', 'rb'))
    elif selected_value == 'Support Vector Classifier':
        loaded_model = pickle.load(open('model_SVM.pkl', 'rb')) 
    elif selected_value == 'K-Nearest Neighbour':
        loaded_model = pickle.load(open('model_KNN.pkl', 'rb')) 
    elif selected_value == 'Ensemble':
        loaded_model = pickle.load(open('model_EN.pkl', 'rb')) 

    # User input fields
    Nitrogen = st.text_input('Nitrogen Content:') 
    Phosphorus = st.text_input('Phosphorus Level:') 
    Potassium = st.text_input('Potassium Value:') 
    Temperature = st.text_input('Temperature Value:') 
    Humidity = st.text_input('Humidity Value:') 
    Ph = st.text_input('PH Value:') 
    Rainfall = st.text_input('Rainfall: ') 

    # Code for prediction
    diagnosis = '' 
    farming_method = ''
    if st.button('Predict'): 
        # Prepare input data
        input_data = [Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall]
        input_data_as_nparray = np.asarray(input_data).astype(float)  # Convert inputs to float
        input_data_reshaped = input_data_as_nparray.reshape(1, -1) 
        
        # Get prediction
        prediction = loaded_model.predict(input_data_reshaped)[0]  # Get predicted crop
        diagnosis = f"Recommended Crop: {prediction}"
        
        # Get farming method
        farming_method = get_farming_method(prediction)
    
    # Display results
    st.success(diagnosis) 
    if farming_method:
        st.info(f"Recommended Farming Method: {farming_method}")

if __name__ == '__main__': 
    main()
