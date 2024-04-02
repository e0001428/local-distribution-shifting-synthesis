from model import Model
import numpy as np
from sklearn.neural_network import MLPClassifier

class MLPModel(Model):
    def __init__(self, seed=1234, hidden_layer_sizes=(100,), max_iter=1000):
        #define model and model name here
        self.seed = seed
        self.model = MLPClassifier(random_state=seed, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
        self.name = "MLP: random state={} hidden_layer_sizes={}".format(seed,hidden_layer_sizes)
