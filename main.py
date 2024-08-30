import base64
import io
import uvicorn
from fastapi import HTTPException, FastAPI, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image
import pytesseract
import re
from starlette.responses import JSONResponse
import base64
import json

from models.skin_type_by_ingredients.ingredients_analysis import get_ingredients_analysis, extract_ingredients_from_text, clean_extracted_text
from models.skin_type_by_face_image.skin_analysis import get_skin_analysis

from pydantic import BaseModel

from utils.users import get_users, write_users, update_user_classification

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class User(BaseModel):
    username: str
    password: str


@app.get("/")
def read_root():
    return {"message": "SkinTell Backend Is Running!"}


@app.post("/skin-analysis/")
async def skin_analysis(file: UploadFile = File(...), username: str = Header(None)):
    image: Image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    image_size = (228, 228)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    img = transform(image).unsqueeze(0).to('cpu')

    img_io, predicted_class = get_skin_analysis(img)

    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    if username:
        update_user_classification(username, predicted_class)

    return JSONResponse(content={
        "predicted_class": predicted_class,
        "heatmap": img_base64
    })


@app.post("/ingredients-list")
async def extract_text_from_image(file: UploadFile = File(...)):
    try:
        # Read the file contents and convert the file contents to an image
        image = Image.open(io.BytesIO(await file.read()))

        image = image.convert("RGB")

        extracted_text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
        # Extract the list of ingredients
        ingredients = await extract_ingredients_from_text(extracted_text)
        cleaned_text = await clean_extracted_text(ingredients)
        print(cleaned_text)

        predicted_skin_types = get_ingredients_analysis(cleaned_text)
        print(predicted_skin_types)

        return JSONResponse(content=predicted_skin_types)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


# Register route to save user data in a list format
@app.post("/register/")
async def register(user: User):
    users = get_users()

    # Check if the username already exists in the list
    for existing_user in users:
        if existing_user['username'] == user.username:
            raise HTTPException(status_code=400, detail="Username already exists")

    # Add the new user to the users list
    users.append({"username": user.username, "password": user.password})

    write_users(users)

    return {"message": "User registered successfully"}


@app.post("/login/")
async def login(user: User):
    users = get_users()

    # Check if the user exists in the users list
    for existing_user in users:
        if existing_user['username'] == user.username and existing_user['password'] == user.password:
            return {"message": "Login successful", "classification": existing_user.get('classification')}

    raise HTTPException(status_code=400, detail="Invalid username or password")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)