import os.path
import re
import torch
import json
from typing import List, Dict
from utils.configs import ROOT_PATH


async def extract_ingredients_from_text(text):
    start_patterns = [
        r"(?i)ingredients(?:\s*\.\.\.|:)?\s*",  # Matches "Ingredients:"
        r"\bwater\b",  # Matches "Water"
        r"\baqua\b"  # Matches "Aqua"
    ]

    # Combine all start patterns into one regex
    start_regex = r"|".join(start_patterns)

    # Try to find the start of the ingredient list
    match = re.search(start_regex, text)

    if match:
        # Extract text starting after the matched pattern
        ingredients_start = match.end()
        ingredients_text = text[ingredients_start:]

        # Optionally, look for an end pattern to stop the extraction
        end_pattern = r">"  # Modify this as needed
        end_match = re.search(end_pattern, ingredients_text)

        if end_match:
            ingredients_text = ingredients_text[:end_match.start()]

        # Clean up the ingredients text
        ingredients_text = ingredients_text.replace("\n", " ").strip()

        return ingredients_text


async def clean_extracted_text(text):
    """Cleans the ingredient text - lowercase, strip spaces and dots, etc...

    Args:
        text: ingredients text from skincare product

    Returns:
        list of ingredients

    """
    text = text.lower()
    # Remove any trailing or leading whitespace
    ingredients_text = text.strip()

    # Split ingredients based on common delimiters (comma, period, semicolon)
    # and clean up extra spaces
    ingredients = re.split(r'[,.:;\n]+', ingredients_text)

    # Strip extra whitespace from each ingredient and filter out empty strings
    ingredients = [ingredient.strip() for ingredient in ingredients if ingredient.strip()]

    return ingredients


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


def get_ingredients_analysis(ingredients_list):
    """Predicts skin type suitability based on products ingredients text

    Args:
        ingredients_list: ingredients list extracted from some skincare product

    Returns:
        The predicted suitable skit types

    """
    model = torch.jit.load(os.path.join(ROOT_PATH, 'models', 'skin_type_by_ingredients', 'ingredients_classifier.pt'))
    model.eval()

    with open(os.path.join(ROOT_PATH, 'models', 'skin_type_by_ingredients', 'ingredient_index_dict.json'), 'r',
              encoding='utf-8') as f:
        ingredient_index_dict = json.load(f)

    with open(os.path.join(ROOT_PATH, 'models', 'skin_type_by_ingredients', 'ingredients_vector_len.json'), 'r',
              encoding='utf-8') as f:
        ingredient_vector_len = json.load(f)["ingredients_vector_len"]

    indexed_ingredients = create_ingredients_vector(ingredients_list, ingredient_index_dict,
                                                    ingredient_vector_len)
    preprocessed_data = indexed_ingredients.long().unsqueeze(0)  # Add batch dimension
    output = model(preprocessed_data)
    skin_types = ['combination', 'dry', 'normal', 'oily', 'sensitive',
                  'pores', 'oiliness', 'dryness', 'dullness', 'uneven',
                  'wrinkles', 'acne', 'blemishes', 'redness', 'dark', 'puffiness']
    predicted_skin_types = [label for prob, label in zip(output[0], skin_types) if prob > 0.5]

    return predicted_skin_types
