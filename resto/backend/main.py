import os
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f"{os.getcwd()}/project-credentials.json"

from typing import Union, Annotated
from fastapi import FastAPI, File, UploadFile

from io import BytesIO
from PIL import Image

from embedding import EmbeddingService

embedding_service = EmbeddingService()
app = FastAPI()


def convertUploadedImageForEmbed(file: UploadFile):
    img = Image.open(file.file).convert("RGB")
    img = img.resize((224, 224))
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


@app.get("/")
def hello_world():
    return "Hello World"


@app.post("/data_search/")
def text_image_search(file: UploadFile, text_data: str | None = None):
    # Max upload size is of 20 MB
    if file.size > 20000000:
        return 413

    if file.content_type not in ("image/jpeg", "image/png"):
        return 406 # Corregir este código
    
    imageBytes = convertUploadedImageForEmbed(file)
    matches = embedding_service.searchImageAndSKU(imageBytes, text_data)

    # Se obtiene la imagen con la mayor probabilidad de cada SKU
    max_similarity = {}
    unique_skus = set()
    for match in matches:
        img, similarity = match
        sku = img.split("_")[0]
        if sku not in unique_skus:
            unique_skus.add(sku)
            max_similarity[sku] = similarity
        else:
            max_similarity[sku] = max(max_similarity[sku], similarity)

    # Se formula a respuesta en JSON.
    response = []
    for k, v in max_similarity.items():
        response.append({ "sku": k, "similarity": v })

    return response


