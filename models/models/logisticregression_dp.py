from diffprivlib.models import LogisticRegression as LRDP
from model import Model
import numpy as np
import math

class LRDPModel(Model):
    def __init__(self, seed=1234, data_norm=0, epsilon=1):
        '''
        data_norm is max L2 norm of training data
          0: take max possible by assuming feature is bounded by [-1,1]
          >0: as per input
          <0: calculated as max L2 in training samples when fit is first called
        '''
        self.seed = seed
        self.model = LRDP(random_state=seed, epsilon=epsilon)
        self.data_norm = data_norm
        self.epsilon = epsilon
        self.seed = seed
        self.name = "LR with DP: random state={} epsilon={}".format(seed,epsilon)

    def train(self, X, y):
        #train self.model with data X/y
        np.random.seed(12+self.seed)
        # due to a bug in model that can't take str classes, make a map here
        d = len(X[0])
        if self.data_norm == 0:
            self.data_norm = math.sqrt(d)
        if self.data_norm >= 0:
            self.model = LRDP(random_state=self.seed, data_norm = self.data_norm, epsilon = self.epsilon)
        else:
            self.model = LRDP(random_state=self.seed, epsilon = self.epsilon)
        self.model.fit(X, y)
        
    def predict(self, X):
        #make prediction
        pred = self.model.predict(X)
        return pred