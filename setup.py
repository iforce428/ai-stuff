import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def train_model():
    """Train the VAE model"""
    print("Training VAE model on MNIST dataset...")
    subprocess.check_call([sys.executable, "train_model.py"])

def run_app():
    """Run the Streamlit app"""
    subprocess.check_call([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    print("🔢 Handwritten Digit Generator Setup")
    print("====================================")
    
    # Install requirements
    print("1. Installing requirements...")
    install_requirements()
    
    # Train model
    print("2. Training VAE model...")
    train_model()
    
    # Run app
    print("3. Starting Streamlit app...")
    run_app() 