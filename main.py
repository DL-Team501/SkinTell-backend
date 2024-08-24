import io
import uvicorn
import easyocr
from fastapi import HTTPException, FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
# import pytesseract
import re
from starlette.responses import JSONResponse

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from models.ingredients_analysis import get_ingredients_analysis
from models.skin_type_by_face_image.skin_analysis import get_skin_analysis

app = FastAPI()

reader = easyocr.Reader(['en'])  # You can specify multiple languages if needed

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "SkinTell Backend Is Running!"}


@app.post("/skin-analysis/")
async def skin_analysis(file: UploadFile = File(...)):
    # Read the image file
    image: Image = Image.open(io.BytesIO(await file.read()))

    class_label = get_skin_analysis(image)

    return class_label


async def extract_ingredients_from_text(text):
    # Search for the word "ingredients" and extract the list after it
    print(text)
    match = re.search(r'(?i)ingredients(?:\s*\.\.\.|:)?\s*(.*?)(?=\.\s|\n\n|$)', text, re.DOTALL)

    # match = re.search(r'ingredients[\s\S]*?:(.*?)(?=\.\s|\.$|\n\n)', text, re.IGNORECASE | re.DOTALL)

    print("match")
    print(match)

    if match:
        # Extract the ingredients list text
        ingredients_text = match.group(1).strip()

        if '.' in ingredients_text:
            # Cut the text until the first period
            ingredients_text = ingredients_text.split('.')[0] + '.'
        print("ingredients_list")
        print(ingredients_text)
        return ingredients_text
    return "No ingredients found."


async def clean_extracted_text(text):
    text = text.replace('\n', ' ')

    # Remove unwanted leading characters like '‘’# but preserve necessary punctuation
    # This will remove '‘’# if they appear at the beginning of any word
    text = re.sub(r"\s['‘’#]+", ' ', text)

    # Remove any multiple spaces that may result from the cleanup
    text = re.sub(r'\s+', ' ', text).strip()

    return text

@app.post("/ingredients-list")
async def extract_text_from_image(file: UploadFile = File(...)):
    try:
        print("HERE!")
        # Read the file contents and convert the file contents to an image
        image = Image.open(io.BytesIO(await file.read()))

        image = image.convert("RGB")
        # Convert the image back to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')  # Save the image as JPEG or appropriate format
        img_byte_arr = img_byte_arr.getvalue()

        # Use easyocr to read text from the image bytes
        extracted_text = reader.readtext(img_byte_arr, detail=0, paragraph=True)
        #
        # # Extract the list of ingredients
        # ingredients = await extract_ingredients_from_text(extracted_text)
        #
        # cleaned_text = await clean_extracted_text(ingredients)
        #
        # print("############################################")
        # print()
        # print(cleaned_text)
        #
        # predicted_skin_types = get_ingredients_analysis(cleaned_text)
        # print(predicted_skin_types)
        # Combine the text results
        text = " ".join(extracted_text)
        print(text)
        predicted_skin_types = get_ingredients_analysis(text)
        print(predicted_skin_types)

        return JSONResponse(content=predicted_skin_types)

        # # Convert the image back to bytes
        # img_byte_arr = io.BytesIO()
        # image.save(img_byte_arr, format='JPEG')  # Save the image as JPEG or appropriate format
        # img_byte_arr = img_byte_arr.getvalue()
        #
        # # Use easyocr to read text from the image bytes
        # result = reader.readtext(img_byte_arr, detail=0, paragraph=True)
        #
        # # Combine the text results
        # text = " ".join(result)
        # print(text)

        # predicted_skin_types = get_ingredients_analysis(text)
        # print(predicted_skin_types)
        #
        # return JSONResponse(content=predicted_skin_types)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/ingredients-list")
# async def extract_text_from_image(file: UploadFile = File(...)):
#     try:
#         print("HERE!")
#         # Read the file contents and convert the file contents to an image
#         image = Image.open(io.BytesIO(await file.read()))
#
#         image = image.convert("RGB")
#
#         # Use Tesseract to extract the text
#         extracted_text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
#
#         # Extract the list of ingredients
#         ingredients = await extract_ingredients_from_text(extracted_text)
#
#         cleaned_text = await clean_extracted_text(ingredients)
#
#         print("############################################")
#         print()
#         print(cleaned_text)
#
#         # # Convert the image back to bytes
#         # img_byte_arr = io.BytesIO()
#         # image.save(img_byte_arr, format='JPEG')  # Save the image as JPEG or appropriate format
#         # img_byte_arr = img_byte_arr.getvalue()
#         #
#         # # Use easyocr to read text from the image bytes
#         # result = reader.readtext(img_byte_arr, detail=0, paragraph=True)
#         #
#         # # Combine the text results
#         # text = " ".join(result)
#         # print(text)
#
#         # predicted_skin_types = get_ingredients_analysis(text)
#         # print(predicted_skin_types)
#         #
#         # return JSONResponse(content=predicted_skin_types)
#     except Exception as e:
#         print(e)
#         raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

