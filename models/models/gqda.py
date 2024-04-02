from model import Model
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

class GQDAModel(Model):
    def __init__(self, seed=1234):
        self.seed = seed
        self.model = QuadraticDiscriminantAnalysis()
        self.name = "Gaussian QDA"