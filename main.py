from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import cv2
import json
import os
import re
import numpy as np
from PIL import Image
import uvicorn

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")
client = genai.configure(api_key=GEMINI_API_KEY)

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
        image_bytes = cv2.imencode(".png", image_array)[1].tobytes()
        
        # Create a GenerativeModel instance
        model = genai.GenerativeModel('gemini-pro-vision')
        
        # Create image part
        image_part = {
            "mime_type": "image/png",
            "data": image_bytes
        }
        
        # Generate content
        response = model.generate_content([prompt, image_part])
        
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
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Process the image
        result = extract_bill_info_gemini(img)
        
        return result
    
    except Exception as e:
        print(f"Error: {str(e)}")  # Log error for debugging
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/health")
def health_check():
    try:
        # Check external service availability
        model = genai.GenerativeModel('gemini-pro-vision')
        return {"status": "healthy", "message": "Server is running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unhealthy: {str(e)}")

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
