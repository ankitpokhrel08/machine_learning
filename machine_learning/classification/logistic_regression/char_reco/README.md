# Handwritten Digit Recognition - Streamlit App

A web application for recognizing handwritten digits (0-9) using a Logistic Regression model trained on the sklearn digits dataset.

## Features

- Interactive drawing canvas for digit input
- Real-time digit prediction with confidence scores
- Probability distribution visualization
- Clean and intuitive user interface

## Model Performance

- **Algorithm:** Logistic Regression with StandardScaler
- **Training Accuracy:** 99.9%
- **Test Accuracy:** 97.2%

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## How to Use

1. Draw a digit (0-9) on the canvas using your mouse or touchscreen
2. Click the "Predict Digit" button
3. View the predicted digit along with confidence score
4. See the probability distribution for all digits
5. Use the clear icon to draw a new digit

## Files

- `app.py` - Main Streamlit application
- `digit_model.pkl` - Trained Logistic Regression model
- `scaler.pkl` - StandardScaler for feature normalization
- `requirements.txt` - Python dependencies

## Technical Details

The model processes drawings by:

1. Converting the canvas drawing to grayscale
2. Resizing to 8x8 pixels (matching training data format)
3. Normalizing pixel values to 0-16 range
4. Scaling features using StandardScaler
5. Making predictions using Logistic Regression

## Dependencies

- streamlit - Web app framework
- numpy - Numerical computations
- scikit-learn - Machine learning model
- opencv-python - Image processing
- Pillow - Image handling
- streamlit-drawable-canvas - Interactive drawing widget
