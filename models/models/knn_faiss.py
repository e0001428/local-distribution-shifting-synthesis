from model import Model
import numpy as np
import faiss

class KNNFAISSModel(Model):
    def __init__(self, seed=1234, k=5, **kwargs):
        self.seed = seed
        self.name = "KNN with FAISS: k={}".format(k)
        self.model = None
        self.k = k
        
    def train(self, X, y):
        dim = len(X[0])
        self.model = faiss.IndexFlatL2(dim)
        self.model.add(np.ascontiguousarray(np.float32(np.array(X))))
        self.y = y

    def predict(self, X):
        dist, ind = self.model.search(np.ascontiguousarray(np.float32(np.array(X))), k=self.k)
        res = [[self.y[i] for i in idx] for idx in ind]
        pdt = [max(set(x), key = x.count) for x in res]
        return pdt