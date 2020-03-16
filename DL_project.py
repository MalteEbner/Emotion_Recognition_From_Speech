from get_data_emotion_recognition import loadData
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Flatten
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np
import pickle


#define dimensions
num_frames = 750
num_feats = 33
num_classes = 4
loadTrueData = True

# Create a sequential model
model = Sequential()
model.add(Conv2D(64, kernel_size=(5,7), strides=(2,4),activation='relu', input_shape=(num_frames,num_feats,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#Load the data
if loadTrueData:
    f = open("data_project", "rb")
    [train_features, train_labels, valid_features, valid_labels, test_features, test_ids] = pickle.load(f)
else:
    noSamples = 20
    train_features = np.ones((noSamples,num_frames,num_feats))
    train_labels = np.random.randint(num_classes, size = noSamples)
    valid_features = np.ones((noSamples,num_frames))
    valid_labels = np.random.randint(num_classes, size = noSamples)

train_labels = to_categorical(train_labels)
valid_labels = to_categorical(valid_labels)
train_features = train_features[..., np.newaxis]
valid_features = valid_features[..., np.newaxis]


from keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, verbose=1),
]
model.fit(train_features,train_labels,  validation_data=(valid_features,valid_labels),callbacks=callbacks)





