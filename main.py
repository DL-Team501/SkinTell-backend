import io
import onnxruntime as ort

from PIL import Image
from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse

app = FastAPI()

# Load the ONNX model
onnx_model_path = "model.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

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
    try:
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
