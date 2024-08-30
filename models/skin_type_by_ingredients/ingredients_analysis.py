import re
import torch
import json
from typing import List, Dict


async def extract_ingredients_from_text(text):
    match = re.search(r'(?i)ingredients(?:\s*\.\.\.|:)?\s*(.*?)(?=\n\n|$)', text, re.DOTALL)

    if match:
        # Extract the ingredients list text
        ingredients_text = match.group(1).strip()

        # Check if there is a period followed by a newline
        match = re.search(r'\.\n', ingredients_text)

        if match:
            # If the match is found, split the text at the position of the first occurrence
            split_position = match.start() + 1  # Position just after the period
            ingredients_text = ingredients_text[:split_position].strip()  # Take the first part of the string up to the period

        return ingredients_text.rstrip('.')
    return "No ingredients found."


async def clean_extracted_text(text):
    text = text.replace('/', ',')
    text = text.replace('!', '')
    text = text.replace(';', ',')
    text = text.replace('\n', ' ')

    # Remove unwanted leading characters like '‘’# but preserve necessary punctuation
    # This will remove '‘’# if they appear at the beginning of any word
    text = re.sub(r"\s['‘’#]+", ' ', text)

    # Remove any multiple spaces that may result from the cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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
    ingredient_list = [re.sub(r'^-.*:', '', ing).strip().lower() for ing in ingredient_list]
    ingredient_list = [re.sub(r'[^a-zA-Z0-9\s]', '', ing).strip() for ing in ingredient_list]

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


def get_ingredients_analysis(ingredients_text):
    """Predicts skin type suitability based on products ingredients text

    Args:
        ingredients_text: ingredients list extracted from some skincare product

    Returns:
        The predicted suitable skit types

    """
    model = torch.jit.load('ingredients_classifier.pt')
    model.eval()

    with open('ingredient_index_dict.json', 'r', encoding='utf-8') as f:
        ingredient_index_dict = json.load(f)

    with open('ingredients_vector_len.json', 'r', encoding='utf-8') as f:
        ingredient_vector_len = json.load(f)["ingredients_vector_len"]

    clean_ingredients_list = clean_ingredients(ingredients_text)
    indexed_ingredients = create_ingredients_vector(clean_ingredients_list, ingredient_index_dict,
                                                    ingredient_vector_len)
    preprocessed_data = indexed_ingredients.long().unsqueeze(0)  # Add batch dimension
    output = model(preprocessed_data)
    skin_types = ['combination', 'dry', 'normal', 'oily', 'sensitive',
                  'pores', 'oiliness', 'dryness', 'dullness', 'uneven',
                  'wrinkles', 'acne', 'blemishes', 'redness', 'dark', 'puffiness']
    predicted_skin_types = [label for prob, label in zip(output[0], skin_types) if prob > 0.5]

    return predicted_skin_types
