from langchain.embeddings import HuggingFaceBgeEmbeddings

from transformers import AutoModel, AutoTokenizer
import torch

class EmbeddingModel:
    def __init__(self, model_path="./local_model", device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        
    def embed_documents(self, documents):
        inputs = self.tokenizer(documents, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings

    def embed_query(self, query):
        inputs = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings
