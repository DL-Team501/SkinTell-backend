import numpy as np
from PIL.Image import Image
from torchvision import transforms

from utils.github import download_model_from_github
from utils.load_model import get_onnx_model

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


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_skin_analysis_by_ingredients(ingredientsStr: str):

    return 1, "class_label"
