from model import Model
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics
from sklearn.utils.extmath import softmax

class NaiveBayesModel(Model):
    # include mix data type
    def __init__(self, seed=1234):
        self.seed = seed
        self.model = [GaussianNB(), CategoricalNB()]
        self.name = "Naive Bayes"
        
    def train(self, x_num, x_cat, y):
        #train self.model with mixed numerical and categorical features
        np.random.seed(12+self.seed)
        if x_num.shape[-1]>0:
            self.model[0].fit(x_num, y)
            self.classes_ = self.model[0].classes_
        if x_cat.shape[-1]>0:
            self.model[1].fit(x_cat, y)
            self.classes_ = self.model[1].classes_
        # mapping class list from model 0 to model 1 (num to cat)
        if x_num.shape[-1]>0 and x_cat.shape[-1]>0:
            self.classes_ = self.model[1].classes_
            self.class_log_prior_ = self.model[1].class_log_prior_
            self.classes_mapping_ = {self.model[0].classes_[i]:i for i in range(len(self.model[0].classes_))}
            self.classes_order_ = [self.classes_mapping_[self.classes_[i]] for i in range(len(self.classes_))]

    def test(self, x_num, x_cat, y):
        prediction = self.predict(x_num, x_cat)
        return metrics.accuracy_score(prediction, y)

    def predict_prob(self, x_num, x_cat):
        if x_num.shape[-1]==0:
            return (softmax(self.model[1]._joint_log_likelihood(x_cat)), self.classes_)
        elif x_cat.shape[-1]==0:
            return (softmax(self.model[0]._joint_log_likelihood(x_num)[:, self.classes_order_]), self.classes_)
        #test self.model with mixed numerical and categorical features
        pred_log_proba = self.model[1]._joint_log_likelihood(x_cat)
        pred_log_proba_num = self.model[0]._joint_log_likelihood(x_num)[:, self.classes_order_]
        pred_log_proba = pred_log_proba + pred_log_proba_num
        pred_log_proba = pred_log_proba - self.class_log_prior_
        return (softmax(pred_log_proba), self.classes_)

    def predict(self, x_num, x_cat):
        pred_proba = self.predict_prob(x_num, x_cat)[0]
        prediction = np.argmax(pred_proba, axis=1)
        prediction = [self.classes_[v] for v in prediction]
        return prediction