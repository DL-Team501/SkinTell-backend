import os
import math
import re
import torch
import gdown
import json
from typing import List, Dict

from utils.load_model import get_onnx_model, to_numpy


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


def apply_positional_encoding(ingredients_vector: torch.Tensor, ingredient_dim: int = 1) -> torch.Tensor:
    """Adds positional encoding to a vector representing a sequence of ingredients.

    Args:
        ingredients_vector: ingredients vector of indexes (each ingredient has index)
        ingredient_dim: vector dimensions of each ingredient

    Returns:
        Ingredients vector of the same shape as the input, with positional encoding added.

    """
    max_ingredients_number = len(ingredients_vector)
    pe = torch.zeros(max_ingredients_number, ingredient_dim)
    position = torch.arange(0, max_ingredients_number, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, ingredient_dim, 2).float() * (-math.log(10000.0) / ingredient_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.squeeze()

    return ingredients_vector + pe


def get_ingredients_analysis(ingredients_text):
    """Predicts skin type suitability based on products ingredients text

    Args:
        ingredients_text: ingredients list extracted from some skincare product

    Returns:
        The predicted suitable skit types

    """
    model_file_name, ingredients_dict_file_name, ingredients_vector_len_file_name = download_model_files()
    model = get_onnx_model(model_file_name)

    with open(ingredients_dict_file_name, 'r', encoding='utf-8') as f:
        ingredient_index_dict = json.load(f)

    with open(ingredients_vector_len_file_name, 'r', encoding='utf-8') as f:
        ingredient_vector_len = json.load(f)["ingredients_vector_len"]

    clean_ingredients_list = clean_ingredients(ingredients_text)
    indexed_ingredients = create_ingredients_vector(clean_ingredients_list, ingredient_index_dict,
                                                    ingredient_vector_len)
    encoded_ingredients = apply_positional_encoding(indexed_ingredients)

    preprocessed_data = encoded_ingredients.unsqueeze(0)  # Add batch dimension
    model_inputs = {model.get_inputs()[0].name: to_numpy(preprocessed_data)}
    model_outs = model.run(None, model_inputs)
    skin_type_probs = model_outs[0][0]
    skin_types = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
    predicted_skin_types = [label for prob, label in zip(skin_type_probs, skin_types) if prob > 0.5]

    return predicted_skin_types


def download_model_files():
    """Downloads from drive all the needed files to run the model

    Returns:
        The paths of the saved files

    """
    model_file_name = 'model.onnx'
    ingredients_dict_file_name = 'ingredient_dict.json'
    ingredients_vector_len_file_name = 'ingredients_vector_len.json'

    if not os.path.isfile(model_file_name):
        model_file_id = '1nChu3VhPRr4kqnI_2-2DymB8SK0u0dza'
        gdrive_url = f'https://drive.google.com/uc?id={model_file_id}'
        gdown.download(gdrive_url, model_file_name, quiet=False)

    if not os.path.isfile(ingredients_dict_file_name):
        ingredient_dict_file_id = '17x_dEmUu3Vtqj-6CP0pl4ITxXa9vzqHn'
        gdrive_url = f'https://drive.google.com/uc?id={ingredient_dict_file_id}'
        gdown.download(gdrive_url, ingredients_dict_file_name, quiet=False)

    if not os.path.isfile(ingredients_vector_len_file_name):
        ingredients_vector_len_file_id = '1OPd8sve1GEQcDIqP5Um3q7teK6eKNFDw'
        gdrive_url = f'https://drive.google.com/uc?id={ingredients_vector_len_file_id}'
        gdown.download(gdrive_url, ingredients_vector_len_file_name, quiet=False)

    return model_file_name, ingredients_dict_file_name, ingredients_vector_len_file_name
