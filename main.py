import io

from PIL import Image
from PIL.Image import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.responses import JSONResponse
import easyocr

from models.skin_analysis import get_skin_analysis
from models.skin_analysis_by_ingredients import get_skin_analysis_by_ingredients

app = FastAPI()

reader = easyocr.Reader(['en'])  # You can specify multiple languages if needed

@app.get("/")
def read_root():
    return {"message": "SkinTell Backend Is Running!"}

@app.post("/ingredients-list")
async def extract_text_from_image(file: UploadFile = File(...)):
    try:
        # Read the file contents and convert the file contents to an image
        image = Image.open(io.BytesIO(await file.read()))

        image = image.convert("RGB")

        result = reader.readtext(io.BytesIO(image), detail=0, paragraph=True)
        
        # Combine the text results
        text = " ".join(result)        

        print(text)        
        class_idx, class_label = get_skin_analysis_by_ingredients(text)

        return JSONResponse(content={"class_id": class_idx, "class_label": class_label})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/skin-analysis/")
async def skin_analysis(file: UploadFile = File(...)):
    # Read the image file
    image: Image = Image.open(io.BytesIO(await file.read()))

    class_idx, class_label = get_skin_analysis(image)

    return JSONResponse(content={"class_id": class_idx, "class_label": class_label})
