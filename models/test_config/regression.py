from models import *

def get_models(seed=1234):
    return [
            BRR(seed=seed),
            KNNRegression(seed=seed, k=20),
            LinearSVRModel(seed=seed),
            LinearReg(seed=seed),
            MLPRTF(seed=seed, hidden_layer_sizes=(64,), epoch=20),
            DTR(seed=seed,max_depth=7),
            RFR(seed=seed,max_depth=7)
        ]
