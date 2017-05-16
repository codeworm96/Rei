from Model import Model

from keras.optimizers import SGD, RMSprop, Adagrad
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

class KerasLSTM(Model):
    # Build and train the model
    def build(self, vocabulary_size, maxlen, x, y):
        print('Build model...')
        self.model = Sequential()
        self.model.add(Embedding(vocabulary_size, 128, input_length=maxlen))
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary", metrics=["accuracy"])
        self.model.fit(x, y, batch_size=32, epochs=10)


    # Load the model from file
    def load(self, filename):
        self.model = load_model("model.h5")

    # Save the model to file
    def save(self, filename):
        self.model.save(filename)

    # predict an array of input
    def predict(self, data):
        return self.model.predict(data)
