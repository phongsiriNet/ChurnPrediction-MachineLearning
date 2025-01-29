from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from joblib import load
import pandas as pd
import numpy as np
import os

# Detailed description of the dataset and columns
description = """
This API allows predictions of customer churn using a machine learning model. The data includes the following attributes:

- **CreditScore**: Customer's credit score
- **Gender**: Gender of the customer (Male/Female)
- **Geography**: The customer's location (France, Germany, Spain)
- **Age**: Age of the customer
- **Tenure**: Duration of the customer's relationship with the bank
- **Balance**: Customer's account balance
- **NumberOfProducts**: Number of products the customer uses
- **HasCrCard**: Whether the customer has a credit card (1 = Yes, 0 = No)
- **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No)
- **EstimatedSalary**: The customer's estimated salary

The goal is to predict whether a customer will churn or not based on these attributes.
"""

app = FastAPI(title="Customer Churn Prediction API",
              description=description,
              version="1.0")

# Define model directory using os.path.join for compatibility across systems
model_dir = os.path.join('c:\\', 'Users', 'Nnz', 'Desktop', 'machine_project', 'models')

@app.get("/")
async def read_root():
    return JSONResponse(content={"message": "Welcome to the Customer Churn Prediction API!"},
                        media_type="application/json",
                        headers={"Content-Encoding": "utf-8"})

@app.post('/prediction', tags=["predictions"])
async def get_prediction(
        CreditScore: int = Query(..., description="Customer's credit score"),
        Gender: str = Query(..., description="Customer's gender (Male/Female)"),
        Geography: str = Query(..., description="Customer's geography (France, Germany, Spain)"),
        Age: int = Query(..., description="Customer's age"),
        Tenure: int = Query(..., description="Customer's tenure (years)"),
        Balance: float = Query(..., description="Customer's balance"),
        NumberOfProducts: int = Query(..., description="Number of products the customer uses"),
        HasCrCard: int = Query(..., description="Does the customer have a credit card (1 = Yes, 0 = No)?"),
        IsActiveMember: int = Query(..., description="Is the customer an active member (1 = Yes, 0 = No)?"),
        EstimatedSalary: float = Query(..., description="Customer's estimated salary")
):
    try:
        # Load the machine learning model, mapping, and columns
        model_path = os.path.join(model_dir, 'best_model_Rb_nobinned.pkl')
        mapping_path = os.path.join(model_dir, 'mapping.pkl')
        columns_path = os.path.join(model_dir, 'columns_Rb_nobinned.pkl')

        # Load model, mapping, and columns using os.path.join
        model = load(model_path)
        mapping = load(mapping_path)
        columns = load(columns_path)

        # Convert Gender using LabelEncoder from mapping
        Gender_encoded = mapping['Gender'][Gender]

        # Prepare input data
        data = {
            'CreditScore': [CreditScore],
            'Gender': [Gender_encoded],
            'Geography': [Geography],  # One-hot encoding will be applied
            'Age': [Age],
            'Tenure': [Tenure],
            'Balance': [Balance],
            'NumberOfProducts': [NumberOfProducts],
            'HasCrCard': [HasCrCard],
            'IsActiveMember': [IsActiveMember],
            'EstimatedSalary': [EstimatedSalary]
        }

        # Convert the input data into a DataFrame
        input_df = pd.DataFrame(data)

        # Apply One-Hot Encoding for Geography
        data_encoded = pd.get_dummies(input_df)

        # Align the input data with the training columns
        data_aligned = data_encoded.reindex(columns=columns, fill_value=0)

        # Predict the probability of class 1 (churn)
        y_pred_proba = model.predict_proba(data_aligned)[:, 1]

        # Set the threshold for prediction
        threshold = 0.4
        y_pred = (y_pred_proba >= threshold).astype(int)

        prediction_label = "Churn" if y_pred[0] == 1 else "No Churn"

        return JSONResponse(content={"prediction": prediction_label},
                            media_type="application/json",
                            headers={"Content-Encoding": "utf-8"})

    except FileNotFoundError as fnf_error:
        return JSONResponse(content={"error": f"File not found: {str(fnf_error)}"},
                            media_type="application/json",
                            headers={"Content-Encoding": "utf-8"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)},
                            media_type="application/json",
                            headers={"Content-Encoding": "utf-8"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5004, reload=True)
