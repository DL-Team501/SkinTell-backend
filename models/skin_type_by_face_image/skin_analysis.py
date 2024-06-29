import os
import numpy as np
import gdown
from PIL.Image import Image
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights

from utils.load_model import get_onnx_model
from utils.configs import ROOT_PATH

output_file_path = os.path.join(ROOT_PATH, "models", "skin_type_by_face_image", "model.onnx")
file_id = '1RVn2gQjFXhevSg8ZBY6SMe-7-rnWSiZH'
model_url = f'https://drive.google.com/uc?id={file_id}'

gdown.download(model_url, output_file_path)

ort_session = get_onnx_model(output_file_path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_skin_analysis(image: Image):
    weights = ViT_B_16_Weights.DEFAULT
    transform = weights.transforms()
    image = transform(image).unsqueeze(0)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
    ort_outs = ort_session.run(None, ort_inputs)

    class_idx = np.argmax(ort_outs[0])
    classes = ['Dark Circle', 'Dry Skin', 'Melasma', 'Normal Skin', 'Oily Skin', 'pustule', 'skin-pore', 'wrinkle']
    label_map = dict(zip(range(0, len(classes)), classes))
    class_label = label_map.get(class_idx)

    return  class_label
