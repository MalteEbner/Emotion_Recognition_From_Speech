import numpy as np
import pickle


##dictionary for translating labels
labelDic = {'angry': 0, 'happy': 1, 'sad': 2, 'neutral': 3}

# Load logmel for training
f = open("data_logmel.train", "rb")
pickle.load(f)  # ignore ids
train_features = pickle.load(f)
train_labels = pickle.load(f)
train_labels = [labelDic[label] for label in train_labels]
f.close()
# load prosody for training
f = open("data_prosody.train", "rb")
pickle.load(f)  # ignore ids
train_features = np.concatenate((train_features, pickle.load(f)), axis=2)
pickle.load(f)  # ignore labels, they are the same
f.close()

# Load logmel for validation
f = open("data_logmel.valid", "rb")
pickle.load(f)  # ignore ids
valid_features = pickle.load(f)
valid_labels = pickle.load(f)
valid_labels = [labelDic[label] for label in valid_labels]
f.close()
# load prosody for validation
f = open("data_prosody.valid", "rb")
pickle.load(f)  # ignore ids
valid_features = np.concatenate((valid_features, pickle.load(f)), axis=2)
pickle.load(f)  # ignore labels, they are the same
f.close()

# Load logmel for testing
f = open("data_logmel.test", "rb")
pickle.load(f)  # ignore ids
test_features = pickle.load(f)
f.close()
# load prosody for testing
f = open("data_prosody.test", "rb")
test_ids = pickle.load(f)  # ignore ids
test_features = np.concatenate((test_features, pickle.load(f)), axis=2)
f.close()

# save in better form
with open("data_project", 'wb') as f_out:
    pickle.dump([train_features, train_labels, valid_features, valid_labels, test_features, test_ids], f_out)

