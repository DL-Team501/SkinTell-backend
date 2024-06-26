import io

import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File

from models.skin_type_by_face_image.skin_analysis import get_skin_analysis

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "SkinTell Backend Is Running!"}


@app.post("/skin-analysis/")
async def skin_analysis(file: UploadFile = File(...)):
    # Read the image file
    image: Image = Image.open(io.BytesIO(await file.read()))

    class_label = get_skin_analysis(image)

    return class_label

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
