from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import BytesIO
from fine import predict_image, generate_pattern
import json

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the input data model for FastAPI
class VideoFile(BaseModel):
    filename: str

@app.get("/generate-pattern/{level}")
async def get_pattern(level: int):
    return generate_pattern(level)

@app.post("/predict-fine")
async def fine_assess_endpoint(file: UploadFile = File(...), colorpattern: str = Form(...)):

    # Save the uploaded video file to a temporary file
    img_bytes = await file.read()
    img_path = f"temp_{file.filename}"

    with open(img_path, "wb") as f:
        f.write(img_bytes)

    print("Image Path:", img_path)

    # Convert colorpattern from string to dictionary
    try:
        colorpattern_dict = json.loads(colorpattern)
        print("Parsed Color Pattern:", colorpattern_dict)
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Invalid colorpattern JSON"}, status_code=400)

    # Assess the video
    result = predict_image(img_path, colorpattern_dict["pattern"])

    # Return the prediction result as a JSON response
    return JSONResponse(content=result)

# test api
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}