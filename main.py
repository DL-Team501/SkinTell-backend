import io

from PIL import Image
from PIL.Image import Image

from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse

from models.skin_analysis import get_skin_analysis

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "SkinTell Backend Is Running!"}


@app.post("/skin-analysis/")
async def skin_analysis(file: UploadFile = File(...)):
    # Read the image file
    image: Image = Image.open(io.BytesIO(await file.read()))

    class_idx, class_label = get_skin_analysis(image)

    return JSONResponse(content={"class_id": class_idx, "class_label": class_label})
