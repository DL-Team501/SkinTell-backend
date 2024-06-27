import numpy as np
from PIL.Image import Image
from torchvision import transforms

from utils.github import download_model_from_github
from utils.load_model import to_numpy, get_onnx_model

github_model_url = "https://github.com/username/repository/raw/main/model.onnx"
local_model_path = "model.onnx"
download_model_from_github(github_model_url, local_model_path)

# Define the transformation to apply to the input image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ort_session = get_onnx_model(local_model_path)


def get_skin_analysis(image: Image):
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

    return class_idx, class_label
