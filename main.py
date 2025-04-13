from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
#import cv2
import json
import os
import re
import numpy as np
from PIL import Image
import uvicorn
import pandas as pd
import dill
from datetime import datetime, timedelta
from calendar import monthrange
import io
from fastapi import Body
from pydantic import BaseModel
from typing import List, Optional

# Update app metadata for Swagger docs
app = FastAPI(
    title="Shaire Backend API",
    description="API for bill extraction and spending prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Bill Extraction

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")
client = genai.Client(api_key=GEMINI_API_KEY)

def extract_bill_info_gemini(image_array):
    if image_array is None:
        return {"error": "No image provided"}
    
    try:
        prompt = """
        You are an expert at extracting information from receipts and bills.
        Analyze the provided image and extract the following information:
        - Merchant name (restaurant or store name)
        - Date of purchase
        - Total amount
        - All items purchased with their individual amounts and types

        Return the information in a JSON format with the following structure:
        {
          "merchant_name": "RESTAURANT NAME",
          "date": "YYYY-MM-DD",
          "total_amount": 45.67,
          "items": [
            {
              "description": "Item name",
              "amount": 12.99,
              "type": "item"
            },
            {
              "description": "Tax",
              "amount": 1.99,
              "type": "tax"
            },
            {
              "description": "Discount",
              "amount": 2.50,
              "type": "discount"
            }
          ]
        }

        For the "type" field, use one of the following values:
        - "item": For regular menu items or products
        - "tax": For tax charges or all other additional fees like service fees
        - "discount": For discounts or promotions (use positive amounts even for discounts)

        If you cannot find some information, use null or empty values. Output ONLY the JSON.
        Ensure the JSON is valid and can be parsed by a computer.
        """
        
        # Convert the image to bytes
        img_byte_arr = io.BytesIO()
        Image.fromarray(image_array).save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        # Use the proper Part helper method
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        
        # Send as contents array
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image_part]
        )
        #response = model.generate_content([prompt, image_part])
        
        json_string = response.text
        
        try:
            data = json.loads(json_string)
            return data
        except json.JSONDecodeError:
            # Cleanup and try again
            json_string = re.sub(r"``````", "", json_string).strip()
            json_match = re.search(r"\{.*\}", json_string, re.DOTALL)
            
            if json_match:
                json_string = json_match.group(0)
                try:
                    data = json.loads(json_string)
                    return data
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON format after cleanup"}
            else:
                return {"error": "No JSON found in response"}
    except Exception as e:
        return {"error": f"Error during Gemini API call: {str(e)}"}

@app.get("/")
def index():
    return {"message": "Bill extraction API is running. Use /extract_bill to upload an image."}

@app.post("/extract_bill")
async def extract_bill(image: UploadFile = File(...)):
    try:
        # Read the image
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Process the image
        result = extract_bill_info_gemini(img)
        
        return result
    
    except Exception as e:
        print(f"Error: {str(e)}")  # Log error for debugging
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get('/health')
def health_check():
    try:
        client.models.get(model="gemini-2.0-flash")
        return {"status": "healthy", "message": "Server is running"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/test")
def test_endpoint():
    return {
        "merchant_name": "TEST RESTAURANT",
        "date": "2023-03-08",
        "total_amount": 45.67,
        "items": [
            {
                "description": "Burger",
                "amount": 12.99,
                "type": "item"
            },
            {
                "description": "Fries",
                "amount": 5.99,
                "type": "item"
            },
            {
                "description": "Soda",
                "amount": 2.49,
                "type": "item"
            },
            {
                "description": "Sales Tax",
                "amount": 1.70,
                "type": "tax"
            },
            {
                "description": "Happy Hour Discount",
                "amount": 2.00,
                "type": "discount"
            }
        ]
    }

# Expense Prediction

class Transaction(BaseModel):
    date: str
    amount: float
    
class PredictionRequest(BaseModel):
    transactions: List[Transaction]
    prediction_days: Optional[int] = 30
    
class SimplePredictionRequest(BaseModel):
    avg_spending: float
    prediction_days: Optional[int] = 30
    
class DailyPrediction(BaseModel):
    date: str
    predicted_amount: float
    
class PredictionResponse(BaseModel):
    total_predicted_spending: float
    average_daily_spending: float
    daily_predictions: List[DailyPrediction]

def get_model_path(filename):
    """Returns absolute path to model file"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "models", filename)

def load_prediction_models():
    """Load preprocessing and prediction models"""
    try:
        with open(get_model_path("preprocessing.pkl"), "rb") as f:
            preprocessing = dill.load(f)
            
        with open(get_model_path("training_prediction.pkl"), "rb") as f:
            prediction_model = dill.load(f)
            
        return preprocessing, prediction_model
    except FileNotFoundError as e:
        print(f"Model file not found: {str(e)}")
        raise Exception(f"Model file not found: {str(e)}")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise Exception(f"Error loading models: {str(e)}")
    

@app.post("/predict_spending", response_model=PredictionResponse)
async def predict_spending(request: PredictionRequest):
    try:
        # Convert transactions to DataFrame
        df = pd.DataFrame([{"date": t.date, "price": t.amount} for t in request.transactions])
        
        # Load models
        prep, train_pred = load_prediction_models()
        
        # Preprocess data
        df, old_mean, new_mean = prep.preprocess(df)
        
        # Generate predictions
        each_day_prediction, total_spending, avg_spent = train_pred.traintest(df)
        
        # Format response
        prediction_days = min(request.prediction_days, len(each_day_prediction))
        
        daily_predictions = []
        for i in range(prediction_days):
            day_date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            daily_predictions.append({
                "date": day_date,
                "predicted_amount": float(each_day_prediction[i])
            })
        
        return {
            "total_predicted_spending": float(total_spending[0]),
            "average_daily_spending": float(avg_spent[0]),
            "daily_predictions": daily_predictions
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_spending_simple", response_model=PredictionResponse)
async def predict_spending_simple(request: SimplePredictionRequest):
    try:
        # Generate synthetic data based on provided average
        synthetic_data = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(10):
            days_ago = 30 - 3*i
            # Create realistic variation in spending
            amount = request.avg_spending * (0.7 + 0.6 * np.random.random())
            synthetic_data.append({
                'date': (base_date + timedelta(days=days_ago)).strftime('%b %d, %Y'),
                'price': amount
            })
            
        df = pd.DataFrame(synthetic_data)
        
        # Load models
        prep, train_pred = load_prediction_models()
        
        # Process and predict
        df, old_mean, new_mean = prep.preprocess(df)
        each_day_prediction, total_spending, avg_spent = train_pred.traintest(df)
        
        # Format response
        prediction_days = min(request.prediction_days, len(each_day_prediction))
        
        daily_predictions = []
        for i in range(prediction_days):
            day_date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            daily_predictions.append({
                "date": day_date,
                "predicted_amount": float(each_day_prediction[i])
            })
        
        return {
            "total_predicted_spending": float(total_spending[0]),
            "average_daily_spending": float(avg_spent[0]),
            "daily_predictions": daily_predictions
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_from_csv")
async def predict_from_csv(csv_file: UploadFile = File(...)):
    try:
        # Read CSV file
        contents = await csv_file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Ensure required columns exist and handle column naming
        if 'date' not in df.columns and 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date'})
            
        if 'amount' not in df.columns and 'price' not in df.columns:
            if 'Amount' in df.columns:
                df = df.rename(columns={'Amount': 'price'})
            elif 'Price' in df.columns:
                df = df.rename(columns={'Price': 'price'})
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="CSV must include either 'date' and 'amount' or 'date' and 'price' columns"
                )
        elif 'amount' in df.columns and 'price' not in df.columns:
            df = df.rename(columns={'amount': 'price'})
        
        # Load models
        prep, train_pred = load_prediction_models()
        
        # Preprocess data
        df, old_mean, new_mean = prep.preprocess(df)
        
        # Generate predictions
        each_day_prediction, total_spending, avg_spent = train_pred.traintest(df)
        
        # Format response
        prediction_days = 30
        
        daily_predictions = []
        for i in range(prediction_days):
            day_date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            daily_predictions.append({
                "date": day_date,
                "predicted_amount": float(each_day_prediction[i]) if i < len(each_day_prediction) else 0.0
            })
        
        return {
            "total_predicted_spending": float(total_spending[0]),
            "average_daily_spending": float(avg_spent[0]),
            "daily_predictions": daily_predictions
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
