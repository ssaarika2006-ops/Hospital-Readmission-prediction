# Hospital Readmission Prediction Using Scikit-learn

## Overview

Hospital Readmission Prediction is a machine learning project that predicts whether a patient may be readmitted to the hospital. The project uses patient and hospital-related information to estimate readmission risk.

## Objective

The main objective is to build a machine learning model that can identify patients who may have a higher probability of hospital readmission based on historical patient information.

## Features Used

The model uses the following features:

* Age
* Length of Stay
* Previous Admissions
* Number of Medications
* Chronic Conditions
* Emergency Visit
* Follow-up Scheduled

## Target Variable

The target variable is `readmitted`.

* 0 = Not Readmitted
* 1 = Readmitted

## Machine Learning Algorithm

The project uses the **Random Forest Classifier** from Scikit-learn.

The model is trained using 80% of the dataset and tested using the remaining 20% of the data.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib
* Streamlit

## Project Structure

```text
Hospital_Readmission_Prediction_Sklearn/
│
├── data/
│   └── hospital_readmission.csv
│
├── models/
│   └── hospital_readmission_model.pkl
│
├── outputs/
│   └── feature_importance.png
│
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
└── HOW_TO_RUN.md
```

## Model Training

The `train_model.py` file:

1. Loads the hospital readmission dataset.
2. Selects the required features.
3. Splits the data into training and testing sets.
4. Trains a Random Forest Classifier.
5. Evaluates the model using accuracy, classification report, and confusion matrix.
6. Saves the trained model as a `.pkl` file.
7. Generates a feature importance chart.

## Prediction

The `predict.py` file loads the trained model and predicts readmission for a sample patient.

It provides:

* Readmission prediction
* Probability of readmission

## Web Application

The `app.py` file creates an interactive Streamlit application where users can enter patient information and evaluate the estimated 30-day readmission risk.

The application also provides:

* Patient risk assessment
* Readmission probability
* Feature importance visualization
* Historical patient data
* Basic care coordination suggestions

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run

### Train the Model

```bash
python train_model.py
```

### Run a Sample Prediction

```bash
python predict.py
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

## Output

The project generates a trained machine learning model and a feature importance visualization. The application displays the predicted readmission classification and probability for the entered patient information.

## Dataset

The dataset used in this project is synthetic and is intended for educational and machine learning practice.

## Limitations

This project is an educational demonstration and should not be used for real clinical diagnosis, treatment, or healthcare decision-making.

## Conclusion

The Hospital Readmission Prediction project demonstrates how machine learning can be used to analyze patient information and predict the possibility of hospital readmission. The Random Forest model, prediction script, feature importance visualization, and Streamlit application provide a complete machine learning workflow.
# Hospital-Readmission-prediction
Hospital Readmission Prediction is a machine learning project that predicts whether a patient is likely to be readmitted to a hospital based on factors such as age, length of stay, previous admissions, medications, chronic conditions, emergency visits, and follow-up scheduling
