import torch
import re
import math
import gdown
import onnx
import json
from typing import List, Dict

from utils.load_model import get_onnx_model


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def clean_ingredients(text: str) -> List[str]:
    """Cleans the ingredient text - lowercase, strip spaces and dots, etc...

    Args:
        text: ingredients text from skincare product

    Returns:
        list of ingredients

    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    ingredient_list = [ingredient.strip() for ingredient in text.split(', ')]
    ingredient_list[-1] = ingredient_list[-1].rstrip('. ')
    ingredient_list[-1] = ingredient_list[-1].rstrip('.')
    ingredient_list = [item for item in ingredient_list if item != ""]

    return ingredient_list


def create_ingredients_vector(ingredient_list: List[str], ingredient_index_dict: Dict[str, int], vector_length: int) \
        -> torch.Tensor:
    """Converts ingredients list to indexes tensor (vector) and adjusts the length to be consistent

    Args:
        ingredient_list: list of ingredients
        ingredient_index_dict: mapping of ingredient to index
        vector_length: length of the ingredients vector (input of the model)

    Returns:
        the ingredients vector of indexes

    """
    ingredient_list_indexes = []
    for ingredient in ingredient_list:
        ingredient_list_indexes.append(ingredient_index_dict.get(ingredient, ingredient_index_dict['<UNK>']))

    if len(ingredient_list) >= vector_length:
        ingredient_list_indexes = ingredient_list_indexes[:vector_length]
    else:
        ingredient_list_indexes = ingredient_list_indexes + [0] * (vector_length - len(ingredient_list_indexes))

    return torch.tensor(ingredient_list_indexes, dtype=torch.float32)


def apply_positional_encoding(ingredients_vector, ingredient_dim=1):
    """

    Args:
        ingredients_vector:
        ingredient_dim:

    Returns:

    """
    max_ingredients_number = len(ingredients_vector)
    pe = torch.zeros(max_ingredients_number, ingredient_dim)
    position = torch.arange(0, max_ingredients_number, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, ingredient_dim, 2).float() * (-math.log(10000.0) / ingredient_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.squeeze()

    return ingredients_vector + pe


def get_ingredients_analysis(ingredients_text, ingredient_vector_len, ingredient_index_dict, model):
    """

    Args:
        ingredients_text:
        ingredient_vector_len:
        ingredient_index_dict:
        model:

    Returns:

    """
    clean_ingredients_list = clean_ingredients(ingredients_text)
    indexed_ingredients = create_ingredients_vector(clean_ingredients_list, ingredient_index_dict,
                                                    ingredient_vector_len)
    encoded_ingredients = apply_positional_encoding(indexed_ingredients)

    preprocessed_data = encoded_ingredients.unsqueeze(0)  # Add batch dimension
    model_inputs = {model.get_inputs()[0].name: to_numpy(preprocessed_data)}
    model_outs = model.run(None, model_inputs)

    return model_outs


def download_model_files():
    # Download model
    model_file_id = '1nChu3VhPRr4kqnI_2-2DymB8SK0u0dza'
    gdrive_url = f'https://drive.google.com/uc?id={model_file_id}'
    gdown.download(gdrive_url, 'model.onnx', quiet=False)

    # Download ingredient dict and ingredients vector len
    ingredient_dict_file_id = '17x_dEmUu3Vtqj-6CP0pl4ITxXa9vzqHn'
    gdrive_url = f'https://drive.google.com/uc?id={ingredient_dict_file_id}'
    gdown.download(gdrive_url, 'ingredient_dict.json', quiet=False)
    ingredients_vector_len_file_id = '1OPd8sve1GEQcDIqP5Um3q7teK6eKNFDw'
    gdrive_url = f'https://drive.google.com/uc?id={ingredients_vector_len_file_id}'
    gdown.download(gdrive_url, 'ingredients_vector_len.json', quiet=False)


if __name__ == '__main__':
    # download_model_files()
    model = get_onnx_model('model.onnx')

    with open('ingredient_dict.json', 'r', encoding='utf-8') as f:
        ingredient_dict = json.load(f)

    with open('ingredients_vector_len.json', 'r', encoding='utf-8') as f:
        ingredients_vector_len = json.load(f)["ingredients_vector_len"]

    ingredients_txt = ("Water/Aqua/Eau, Glycerin, Niacinamide, Butylene Glycol, Ectoin, Sodium Hyaluronate, Hydrolyzed "
                       "Sodium Hyaluronate, Sodium Acetylated Hyaluronate, Sodium Hyaluronate Crosspolymer, "
                       "Acetyl Hexapeptide-8, Palmitoyl Tripeptide-1, Palmitoyl Tetrapeptide-7, Oligopeptide-2, "
                       "Saccharide Isomerate, Camellia Sinensis (Green Tea) Leaf Extract, Camellia Sinensis (White "
                       "Tea) Leaf Extract, Centella Asiatica Extract, Vitis Vinifera (Grape) Seed Extract, "
                       "Sodium Carboxymethyl Beta-Glucan, Bakuchiol, Inonotus Obliquus (Mushroom) Extract, "
                       "Tetrahydrocurcumin, Ceramide NP, Ceramide AP, Ceramide EOP, Phospholipids, Sphingolipids, "
                       "Phytosphingosine, Colloidal Oatmeal, Bisabolol, Superoxide Dismutase, Terminalia "
                       "Ferdinandiana (Kakadu Plum) Fruit Extract, Sodium PCA, Glutathione, Squalane, Caffeine, "
                       "Panthenol, Butyrospermum Parkii (Shea) Butter, Aloe Barbadensis Leaf Juice, Arnica Montana "
                       "Flower Extract, Acetyl Glutamine, Bifida Ferment Lysate, Chamomilla Recutita (Matricaria) "
                       "Flower Extract, Withania Somnifera (Ashwagandha) Root Extract, Glycolic Acid, Mandelic Acid, "
                       "Lactobionic Acid, Ethylhexylglycerin, Coco-Glucoside, Lauryl Glucoside, Decyl Glucoside, "
                       "Polyglyceryl-10 Laurate, Diheptyl Succinate, Dimethicone, Pentylene Glycol, Sodium Lauroyl "
                       "Lactylate, PVM/MA Decadiene Crosspolymer, VP/VA Copolymer, Xanthan Gum, Potassium Hydroxide, "
                       "Polysorbate 20, Dimethicone Crosspolymer, Carbomer, Cholesterol, Capryloyl Glycerin/Sebacic "
                       "Acid Copolymer, Leuconostoc/Radish Root Ferment Filtrate, Sodium Citrate, Citric Acid, "
                       "Sodium Metasilicate, Gluconolactone, Tetrasodium Glutamate Diacetate, Caprylic/Capric "
                       "Triglyceride, Jasminum Officinale (Jasmine) Flower/Leaf Extract, Vitis Vinifera (Grape) Fruit "
                       "Extract, Eugenia Caryophyllus (Clove) Flower Extract, Lavandula Angustifolia (Lavender) "
                       "Flower/Leaf/Stem Extract, Raspberry Ketone, Phenoxyethanol, Potassium Sorbate.")

    skin_type_probs = get_ingredients_analysis(ingredients_txt, ingredients_vector_len, ingredient_dict, model)[0][0]
    skin_types = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
    predicted_skin_types = [label for prob, label in zip(skin_type_probs, skin_types) if prob > 0.5]
    print(predicted_skin_types)
