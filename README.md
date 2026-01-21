# 🏠 House Price Prediction System

A machine learning-powered web application that predicts house prices based on key property features using the Random Forest Regressor algorithm.

## 📋 Project Overview

This project implements a complete end-to-end machine learning system for predicting house sale prices using the "House Prices: Advanced Regression Techniques" dataset.

### Features Used (6 out of 9 recommended)
1. **OverallQual** - Overall material and finish quality (1-10)
2. **GrLivArea** - Above grade living area in square feet
3. **TotalBsmtSF** - Total square feet of basement area
4. **GarageCars** - Size of garage in car capacity
5. **YearBuilt** - Original construction date
6. **Neighborhood** - Physical location (categorical)

### Algorithm
- **Random Forest Regressor** with 200 estimators
- Model persistence using **Joblib**
- Feature scaling with StandardScaler
- Label encoding for categorical variables

### Performance Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 🗂️ Project Structure

```
HousePrice_Project_yourName_matricNo/
│
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── HousePrice_hosted_webGUI_link.txt  # Submission information
├── README.md                          # This file
│
├── model/
│   ├── model_building.ipynb           # Model development notebook
│   ├── house_price_model.pkl          # Trained Random Forest model
│   ├── feature_scaler.pkl             # StandardScaler object
│   └── neighborhood_encoder.pkl       # LabelEncoder for neighborhoods
│
└── (optional: static/ and templates/ for Flask)
```

---

## 🚀 Installation & Local Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <your-github-repo-url>
cd HousePrice_Project_yourName_matricNo
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# On Linux/Mac
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the Model (First Time Only)
Open and run the Jupyter notebook to train and save the model:
```bash
jupyter notebook model/model_building.ipynb
```

Run all cells in the notebook. This will generate:
- `model/house_price_model.pkl`
- `model/feature_scaler.pkl`
- `model/neighborhood_encoder.pkl`

### Step 5: Run the Web Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 🌐 Deployment Instructions

### Option 1: Streamlit Cloud (Recommended - FREE)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - House Price Prediction"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Important Notes:**
   - Ensure all model files (.pkl) are committed to GitHub
   - The model files must be in the `model/` directory
   - Your app will be live at: `https://your-app-name.streamlit.app`

### Option 2: Render.com (FREE)

1. **Create `render.yaml` in root directory:**
```yaml
services:
  - type: web
    name: house-price-predictor
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

2. **Deploy:**
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Render will auto-detect the configuration
   - Click "Create Web Service"

### Option 3: PythonAnywhere (FREE Tier Available)

1. **Upload files:**
   - Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
   - Upload your project files via Files tab

2. **Install dependencies:**
   ```bash
   pip install --user -r requirements.txt
   ```

3. **Create web app:**
   - Go to Web tab
   - Create new web app
   - Choose Flask
   - Configure WSGI file to run Streamlit

**Note:** PythonAnywhere is better suited for Flask. For Streamlit, use Streamlit Cloud or Render.

### Option 4: Vercel (Requires Additional Configuration)

Vercel is optimized for Next.js and static sites. For Python applications like Streamlit, use Streamlit Cloud or Render instead.

---

## 🧪 Testing the Application

### Local Testing
1. Start the application: `streamlit run app.py`
2. Navigate to `http://localhost:8501`
3. Enter house features:
   - Overall Quality: 7
   - Living Area: 2000 sq ft
   - Basement Area: 1000 sq ft
   - Garage Capacity: 2 cars
   - Year Built: 2005
   - Neighborhood: Choose any
4. Click "Predict House Price"
5. Verify prediction is displayed

### Model Verification
Open the Jupyter notebook and run all cells to verify:
- Data loading works correctly
- Model training completes without errors
- Evaluation metrics are calculated
- Model files are saved successfully
- Model can be reloaded and used for predictions

---

## 📊 Model Development Details

### Data Preprocessing
1. **Missing Value Handling:**
   - Numerical features: Filled with median
   - Categorical features: Filled with mode

2. **Feature Encoding:**
   - Neighborhood encoded using LabelEncoder

3. **Feature Scaling:**
   - StandardScaler applied to all features
   - Ensures equal contribution from all features

### Model Training
- **Train-Test Split:** 80-20
- **Algorithm:** Random Forest Regressor
- **Hyperparameters:**
  - n_estimators: 200
  - max_depth: 15
  - min_samples_split: 5
  - min_samples_leaf: 2
  - random_state: 42

### Evaluation
The model is evaluated using:
- **MAE** - Average absolute prediction error
- **MSE** - Average squared prediction error
- **RMSE** - Standard deviation of prediction errors
- **R² Score** - Proportion of variance explained

---

## 📁 GitHub Repository Setup

### Repository Name
`HousePrice_Project_yourName_matricNo`

Example: `HousePrice_Project_JohnDoe_U20CS1234`

### README Badge (Optional)
Add a Streamlit badge to your README:
```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
```

### .gitignore File
Create a `.gitignore` file to exclude unnecessary files:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Don't ignore model files - they need to be deployed!
# model/*.pkl
```

**Important:** Do NOT add `*.pkl` to .gitignore as the model files must be deployed!

---

## 🐛 Troubleshooting

### Model Loading Error
**Error:** "Error loading model: [Errno 2] No such file or directory"

**Solution:**
- Ensure model files are in the `model/` directory
- Run the Jupyter notebook to generate model files
- Verify files exist: `house_price_model.pkl`, `feature_scaler.pkl`, `neighborhood_encoder.pkl`

### Import Errors
**Error:** "ModuleNotFoundError: No module named 'streamlit'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Prediction Errors
**Error:** "ValueError: could not convert string to float"

**Solution:**
- Ensure all input values are within valid ranges
- Check that neighborhood is properly encoded
- Verify scaler and encoder are loaded correctly

### Port Already in Use
**Error:** "OSError: [Errno 48] Address already in use"

**Solution:**
```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or specify a different port
streamlit run app.py --server.port 8502
```

---

## 📝 Submission Checklist

Before submitting, ensure:

- [ ] Model training notebook runs without errors
- [ ] All evaluation metrics are calculated and displayed
- [ ] Model files (.pkl) are generated and saved
- [ ] Web application runs locally without errors
- [ ] Application successfully loads the model
- [ ] Predictions work correctly with sample inputs
- [ ] Application is deployed to cloud platform
- [ ] GitHub repository is created with proper structure
- [ ] README.md is complete and informative
- [ ] requirements.txt includes all dependencies
- [ ] HousePrice_hosted_webGUI_link.txt is filled out completely
- [ ] All files are committed and pushed to GitHub
- [ ] Deployed application URL is accessible
- [ ] Project uploaded to Scorac before deadline

---

## 👥 Team Information

**Name:** [Your Full Name]  
**Matric Number:** [Your Matric Number]  
**Course:** [Course Code]  
**Submission Date:** January 22, 2026

---

## 📄 License

This project is submitted as part of academic coursework.

---

## 🙏 Acknowledgments

- Dataset: House Prices - Advanced Regression Techniques (Kaggle)
- Framework: Streamlit for rapid web app development
- ML Library: Scikit-learn for machine learning algorithms
- Visualization: Matplotlib and Seaborn for data analysis

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
3. Check Scikit-learn documentation: [scikit-learn.org](https://scikit-learn.org)

---

**Good luck with your submission! 🚀**
