from models import *

def get_models(seed=1234):
    return [
            NaiveBayesModel(seed=seed),
            KNNFAISSModel(seed=seed, k=5),
            KNNFAISSModel(seed=seed, k=20),
            LinearSVCModel(seed=seed),
            LinearSVCSTACKINGModel(seed=seed),
            LRModel(seed=seed),
            MLPTFModel(seed=seed, hidden_layer_sizes=(64,), epoch=20),
            DecisionTreeModel(seed=seed,max_depth=3),
            DecisionTreeModel(seed=seed,max_depth=5),
            DecisionTreeModel(seed=seed,max_depth=7),
            RFModel(seed=seed,max_depth=3),
            RFModel(seed=seed,max_depth=5),
            RFModel(seed=seed,max_depth=7)
        ]
