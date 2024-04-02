from model import Model
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

class DecisionTreeModel(Model):
    def __init__(self, seed=1234, max_depth=5):
        self.seed = seed
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
        self.name = "Decision Tree: random state={} max_depth={}".format(seed,max_depth)

    def train(self, x, y):
        #train self.model with data x/y
        np.random.seed(12+self.seed)
        self.model.fit(x,y)