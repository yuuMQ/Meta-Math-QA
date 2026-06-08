from sentence_transformers import SentenceTransformer
import torch
from inspect_data import InspectData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MetaMathEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-large",
            device=device,
        )

    def embed(self, texts):
        prefixed = ['passage: {}'.format(t for t in texts)]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, query):
        embedding = self.model.encode(
            'query: {}'.format(query),
            normalize_embeddings=True,
        )
        return embedding.tolist()
