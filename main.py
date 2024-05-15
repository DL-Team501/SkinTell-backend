import io
import os

import onnxruntime as ort
import requests

from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse
from torchvision import transforms

app = FastAPI()

load_dotenv()

github_token = os.getenv("GITHUB_TOKEN")

if github_token is None:
    raise ValueError("GitHub token not found in environment variables.")

headers = {
    "Authorization": f"token {github_token}"
}


# Function to download the ONNX model from GitHub
def download_model_from_github(github_url, model_path):
    response = requests.get(github_url, headers=headers)

    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception("Failed to download model from GitHub")


# Define the URL of the ONNX model on GitHub
github_model_url = "https://github.com/username/repository/raw/main/model.onnx"
# Update the GitHub URL with your repository details

# Define the local path where the model will be downloaded
local_model_path = "model.onnx"

# Download the model from GitHub
download_model_from_github(github_model_url, local_model_path)

# Load the ONNX model
ort_session = ort.InferenceSession(local_model_path)

# Define the transformation to apply to the input image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.get("/")
def read_root():
    return {"message": "SkinTell Backend Is Running!"}


@app.post("/skin-analysis/")
async def skin_analysis(file: UploadFile = File(...)):
    # Read the image file
    image = Image.open(io.BytesIO(await file.read()))

    # Preprocess the image
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Run the image through the ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
    ort_outs = ort_session.run(None, ort_inputs)

    # Get the predicted class
    class_idx = np.argmax(ort_outs[0])

    # Map the class index to a human-readable label
    labels = {0: 'class_0', 1: 'class_1'}  # Update with your actual labels
    class_label = labels.get(class_idx, "Unknown")

    return JSONResponse(content={"class_id": class_idx, "class_label": class_label})
