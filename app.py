"""
House Price Prediction Web Application
PART B: Web GUI Application using Streamlit

This application loads a pre-trained Random Forest model and allows users
to input house features to predict the sale price.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f8ff;
        border: 2px solid #4CAF50;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin: 20px 0;
    }
    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_artifacts():
    """Load the trained model, scaler, and encoder"""
    try:
        model_path = os.path.join('model', 'house_price_model.pkl')
        scaler_path = os.path.join('model', 'feature_scaler.pkl')
        encoder_path = os.path.join('model', 'neighborhood_encoder.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        
        return model, scaler, encoder, None
    except Exception as e:
        return None, None, None, str(e)

def get_neighborhood_options(encoder):
    """Get the list of neighborhoods from the encoder"""
    try:
        return list(encoder.classes_)
    except:
        # Fallback list of common neighborhoods
        return ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 
                'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
                'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown',
                'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker']

def predict_price(model, scaler, encoder, input_data):
    """Make price prediction based on user input"""
    try:
        # Create DataFrame with feature names
        feature_names = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt', 'Neighborhood_Encoded']
        
        # Encode neighborhood
        neighborhood_encoded = encoder.transform([input_data['Neighborhood']])[0]
        
        # Prepare input array
        input_array = np.array([[
            input_data['OverallQual'],
            input_data['GrLivArea'],
            input_data['TotalBsmtSF'],
            input_data['GarageCars'],
            input_data['YearBuilt'],
            neighborhood_encoded
        ]])
        
        # Scale the features
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        return prediction, None
    except Exception as e:
        return None, str(e)

def main():
    # Header
    st.title("🏠 House Price Prediction System")
    st.markdown("### Predict house prices using machine learning")
    st.markdown("---")
    
    # Load model and artifacts
    model, scaler, encoder, error = load_model_and_artifacts()
    
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("Please ensure the model files are in the 'model' directory:")
        st.code("- model/house_price_model.pkl\n- model/feature_scaler.pkl\n- model/neighborhood_encoder.pkl")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application predicts house sale prices based on:
        
        **Features:**
        - Overall Quality
        - Living Area (sq ft)
        - Basement Area (sq ft)
        - Garage Capacity
        - Year Built
        - Neighborhood
        
        **Algorithm:**
        Random Forest Regressor
        
        **Model Persistence:**
        Joblib
        """)
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("""
        1. Enter house features in the form
        2. Click 'Predict Price' button
        3. View the predicted sale price
        """)
    
    # Main content area
    st.header("📝 Enter House Details")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Property Characteristics")
        
        overall_qual = st.slider(
            "Overall Quality (1-10)",
            min_value=1,
            max_value=10,
            value=5,
            help="Overall material and finish quality of the house"
        )
        
        gr_liv_area = st.number_input(
            "Above Grade Living Area (sq ft)",
            min_value=300,
            max_value=6000,
            value=1500,
            step=50,
            help="Living area above ground level in square feet"
        )
        
        total_bsmt_sf = st.number_input(
            "Total Basement Area (sq ft)",
            min_value=0,
            max_value=6000,
            value=1000,
            step=50,
            help="Total square feet of basement area"
        )
    
    with col2:
        st.subheader("Additional Features")
        
        garage_cars = st.selectbox(
            "Garage Capacity (number of cars)",
            options=[0, 1, 2, 3, 4],
            index=2,
            help="Size of garage in car capacity"
        )
        
        year_built = st.slider(
            "Year Built",
            min_value=1872,
            max_value=2026,
            value=2000,
            help="Original construction date"
        )
        
        neighborhood_options = get_neighborhood_options(encoder)
        neighborhood = st.selectbox(
            "Neighborhood",
            options=sorted(neighborhood_options),
            help="Physical location within Ames city limits"
        )
    
    st.markdown("---")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button("🔮 Predict House Price")
    
    # Make prediction when button is clicked
    if predict_button:
        # Prepare input data
        input_data = {
            'OverallQual': overall_qual,
            'GrLivArea': gr_liv_area,
            'TotalBsmtSF': total_bsmt_sf,
            'GarageCars': garage_cars,
            'YearBuilt': year_built,
            'Neighborhood': neighborhood
        }
        
        # Show spinner while predicting
        with st.spinner('Calculating prediction...'):
            predicted_price, error = predict_price(model, scaler, encoder, input_data)
        
        if error:
            st.error(f"❌ Prediction error: {error}")
        else:
            # Display prediction
            st.markdown("### 🎯 Prediction Result")
            st.markdown(f"""
                <div class="prediction-box">
                    Estimated House Price: ${predicted_price:,.2f}
                </div>
            """, unsafe_allow_html=True)
            
            # Display input summary
            st.markdown("### 📊 Input Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Quality", f"{overall_qual}/10")
                st.metric("Garage Capacity", f"{garage_cars} cars")
            
            with col2:
                st.metric("Living Area", f"{gr_liv_area:,} sq ft")
                st.metric("Year Built", year_built)
            
            with col3:
                st.metric("Basement Area", f"{total_bsmt_sf:,} sq ft")
                st.metric("Neighborhood", neighborhood)
            
            # Additional insights
            st.markdown("---")
            st.markdown("### 💡 Insights")
            
            price_per_sqft = predicted_price / gr_liv_area if gr_liv_area > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📐 **Price per Square Foot:** ${price_per_sqft:.2f}")
            with col2:
                house_age = 2026 - year_built
                st.info(f"⏰ **House Age:** {house_age} years")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>House Price Prediction System | Powered by Random Forest Algorithm</p>
            <p style='font-size: 12px;'>Built with Streamlit & Scikit-learn</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
