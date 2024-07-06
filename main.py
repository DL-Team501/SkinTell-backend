import io
import uvicorn
import easyocr
from fastapi import HTTPException, FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.responses import JSONResponse

from models.ingredients_analysis import get_ingredients_analysis
from models.skin_type_by_face_image.skin_analysis import get_skin_analysis

app = FastAPI()

reader = easyocr.Reader(['en'])  # You can specify multiple languages if needed


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

        return JSONResponse(content={"class_id": 1, "class_label": "ala"})
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
