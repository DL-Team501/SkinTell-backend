import io
import uvicorn
import easyocr
from torchvision import transforms
from fastapi import HTTPException, FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.responses import JSONResponse
import base64
import json

from models.ingredients_analysis import get_ingredients_analysis
from models.skin_type_by_face_image.skin_analysis import get_skin_analysis
from pathlib import Path
from pydantic import BaseModel

app = FastAPI()

reader = easyocr.Reader(['en'])  # You can specify multiple languages if needed

USERS_FILE = Path("users.json")

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
async def skin_analysis(file: UploadFile = File(...)):
    # Read the image file
    image: Image = Image.open(io.BytesIO(await file.read()))

    class_label = get_skin_analysis(image)

    return class_label


@app.post("/ingredients-list")
async def extract_text_from_image(file: UploadFile = File(...)):
    try:
        # Read the file contents and convert the file contents to an image
        image = Image.open(io.BytesIO(await file.read()))

        image = image.convert("RGB")
        # Convert the image back to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')  # Save the image as JPEG or appropriate format
        img_byte_arr = img_byte_arr.getvalue()

        # Use easyocr to read text from the image bytes
        result = reader.readtext(img_byte_arr, detail=0, paragraph=True)

        # Combine the text results
        text = " ".join(result)
        print(text)

        predicted_skin_types = get_ingredients_analysis(text)
        print(predicted_skin_types)

        return JSONResponse(content=predicted_skin_types)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

# Register route to save user data in a list format
@app.post("/register/")
async def register(user: User):
    print("a")
    try:
        # Load the existing users from the file, if it exists
        if USERS_FILE.exists():
            print("b")
            with USERS_FILE.open("r") as f:
                users = json.load(f)
        else:
            users = []
        print(users)
        # Check if the username already exists in the list
        for existing_user in users:
            if existing_user['username'] == user.username:
                raise HTTPException(status_code=400, detail="Username already exists")
        print("dd")
        # Add the new user to the users list
        users.append({"username": user.username, "password": user.password})

        # Save the updated users list back to the JSON file
        with USERS_FILE.open("w") as f:
            json.dump(users, f, indent=4)

        return {"message": "User registered successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login/")
async def login(user: User):
    try:
        # Load the existing users from the file
        if USERS_FILE.exists():
            with USERS_FILE.open("r") as f:
                users = json.load(f)
        else:
            raise HTTPException(status_code=400, detail="No users found")

        # Check if the user exists in the users list
        for existing_user in users:
            if existing_user['username'] == user.username and existing_user['password'] == user.password:
                return {"message": "Login successful"}

        raise HTTPException(status_code=400, detail="Invalid username or password")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)

