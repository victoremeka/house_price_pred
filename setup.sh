#!/bin/bash

# House Price Prediction System - Setup Script
# This script helps you set up the project quickly

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   House Price Prediction System - Automated Setup             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 detected: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                     NEXT STEPS                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "1. Train the model:"
echo "   jupyter notebook model/model_building.ipynb"
echo "   (Run all cells to generate model files)"
echo ""
echo "2. Run the web application:"
echo "   streamlit run app.py"
echo ""
echo "3. Access the app at:"
echo "   http://localhost:8501"
echo ""
echo "For detailed instructions, see QUICKSTART.txt or README.md"
echo ""
