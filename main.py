from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fine import predict_image, generate_pattern
import json
import os

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Save images in temp directory
TEMP_DIR = "temp_dir"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/generate-pattern/{level}")
async def get_pattern(level: int):
    return generate_pattern(level)

@app.post("/predict-fine")
async def fine_assess_endpoint(file: UploadFile = File(...), colorpattern: str = Form(...)):

    # ensure temp_dir exists
    os.makedirs(TEMP_DIR, exist_ok=True)

    # define img path
    img_path = os.path.join(TEMP_DIR, f"temp_{file.filename}")

    # save uploaded img file to a temp folder
    img_bytes = await file.read()
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    # convert colorpattern from string to dictionary
    try:
        colorpattern_dict = json.loads(colorpattern)
        print("Parsed Color Pattern:", colorpattern_dict)
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Invalid colorpattern JSON"}, status_code=400)

    # assess image
    result = predict_image(img_path, colorpattern_dict["pattern"])

    # cleanup - delete the temp image file
    os.remove(img_path)

    # return prediction result
    return JSONResponse(content=result)

# test api
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}