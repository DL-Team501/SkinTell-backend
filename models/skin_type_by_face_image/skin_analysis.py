import os
import gdown
import torch
from PIL import Image
from models.skin_type_by_face_image.SkinNet import SkinNet
from models.skin_type_by_face_image.gradCAM import get_gradcam_heatmap, generate_gradcam_image
from utils.configs import ROOT_PATH

output_file_path = os.path.join(ROOT_PATH, "models", "skin_type_by_face_image", "model.pth")
file_id = '1jXI_Sm2IzeAxbkGnw639KU_OQQIdMWsx'
model_url = f'https://drive.google.com/uc?id={file_id}'

gdown.download(model_url, output_file_path)


def get_skin_analysis(image: Image):
    classes = ['acne','dark circle', 'oil', 'dry', 'wrinkle']
    loaded_model = SkinNet(num_classes=len(classes))
    loaded_model.load_state_dict(torch.load(output_file_path, map_location=torch.device('cpu')))
    loaded_model.to('cpu')
    loaded_model.eval()

    target_layer = loaded_model.features[-1]
    heatmap, predicted_class_idx = get_gradcam_heatmap(loaded_model, image, target_layer)

    return generate_gradcam_image((image.shape[2], image.shape[3]), heatmap, predicted_class_idx, classes)

