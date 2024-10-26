from vertexai.vision_models import MultiModalEmbeddingModel, Image
from vertexai.language_models import TextEmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import json
from typing import Optional
import numpy as np

class EmbeddingService:
    def __init__(self):
        self.model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
        self.text_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        with open('embeddings/embeddings.json', 'r') as file:
            self.embeddings = json.load(file)
        epsilon = 0.20
        dbscan = DBSCAN(eps=epsilon, metric='euclidean', min_samples=1)
        embeddings = [images[image]["image_embedding"] for image in images]
        classes = dbscan.fit_predict(embeddings)

    # TODO: Implementar ANN (clusters)
    def searchImageAndSKU(self, imageBase64: bytes, text_data: Optional[str] = None):
        embedded_img = self.__generateImageEmbeddings__(imageBase64)
        embedded_text = self.__generateTextEmbeddings__(text_data)

        matches = []
        for imagefile in self.embeddings:
            # Image similarity
            similitud = cosine_similarity(
                [embedded_img],
                [self.embeddings[imagefile]["image_embedding"]]
            ).item()
            if similitud >= 0.8:
                matches.append((imagefile, similitud))
                continue
            
            # Text similarity
            if text_data != None:
                text_similitud = cosine_similarity(
                    [embedded_text],
                    [self.embeddings][imagefile]["text_embedding"]
                ).item()
                if text_similitud >= 0.8:
                    matches.append((imagefile, similitud))
        
        return matches


    def __generateTextEmbeddings__(self, text: str = None):
        if text == None: return None

        embedded_text = self.text_model.get_embeddings([text], output_dimensionality=128,)
        return embedded_text[0].values


    def __generateImageEmbeddings__(self, imageBase64: bytes):
        img_part = Image(imageBase64)
        embedded_img = self.model.get_embeddings(image=img_part, dimension=128,)
        return embedded_img.image_embedding