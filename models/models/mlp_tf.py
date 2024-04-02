from model import Model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from utils.model_utils import gpu_selection, mixed_up
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras.utils import np_utils

class MLPTFModel(Model):
    def __init__(self, seed=1234, hidden_layer_sizes=(100,), epoch=20, mixedup=False):
        #define model and model name here
        self.seed = seed
        self.epoch = epoch
        self.hls = hidden_layer_sizes
        self.mixedup = mixedup
        self.name = "MLP TF: random state={} hidden_layer_sizes={}{}".format(seed,hidden_layer_sizes," with mixedup" if mixedup else "")

    def train(self, X, y):
        #train self.model with data X/y
        np.random.seed(12+self.seed)
        from tensorflow.python.keras import backend as K
        
        '''
        gpu = gpu_selection()
        if gpu=="-1":
            # commented out due to current imcapable of changing gpu option within same process, hence use cpu only
            config = tf.compat.v1.ConfigProto(device_count = {'CPU': 1, 'GPU': 0})
            config.gpu_options.visible_device_list = ""
        else:
            config = tf.compat.v1.ConfigProto(device_count = {'CPU': 1, 'GPU': 1})
            config.gpu_options.visible_device_list = gpu
        '''
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        sess = tf.compat.v1.Session(config=config)
        K.set_session(sess)
        
        tf.random.set_seed(self.seed)
        # label encoding
        self.encoder = LabelEncoder()
        self.encoder.fit(y)
        y_encoded = self.encoder.transform(y)
        y_keras = np_utils.to_categorical(y_encoded)
        n_classes = len(self.encoder.classes_)
        
        # set model structure as per input shape
        self.model = Sequential()
        self.model.add(Dense(self.hls[0], activation='relu', input_shape=(len(X[0]),)))
        for l in self.hls[1:]:
              self.model.add(Dense(l, activation='relu'))
        self.model.add(Dense(n_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
        if self.mixedup:
            for ep in range(self.epoch):
                #print("epoch #{}:".format(ep))
                X_mixedup, y_mixedup = mixed_up(X, y_keras)
                self.model.fit(X_mixedup, y_mixedup, batch_size=32, epochs=1, verbose=0)
        else:
            self.model.fit(X, y_keras, batch_size=32, epochs=self.epoch, verbose=0)
        
    def predict(self, X):
        #make prediction
        pdt = self.model.predict(X)
        return self.encoder.inverse_transform(np.argmax(pdt, axis=1))
