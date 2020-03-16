import xarray as xr
import numpy
import pickle

def loadData():
    directory = "/Users/malteebner/Library/Mobile Documents/com~apple~CloudDocs/9. Semester/DL for NLP/Project/emotion_recognition/"  # exchange with the path to the data you have downloaded from Ilias

    names = ['prosody.train', 'logMel.train', 'prosody.valid', 'logMel.valid', 'prosody.test', 'logMel.test']
    file_paths = [directory + "data_prosody.train", directory + "data_logMel.train", directory + "data_prosody.valid", directory + "data_logMel.valid", directory + "data_prosody.test", directory + "data_logMel.test"]

    for file_path, name in zip(file_paths, names):
        print('opening ' + file_path)
        data = xr.open_dataset(file_path)

        # The dataset looks like this:
        #
        # <xarray.Dataset>
        # Dimensions:         (feature_count: 7, instance: 10039, time: 750)
        # Dimensions without coordinates: feature_count, instance, time
        # Data variables:
        #     file_name       (instance) object ...
        #     feature_names   (feature_count) object ...
        #     feature_value   (instance, time, feature_count) float32 ...
        #     speaker_gender  (instance) object ...
        #     speech_type     (instance) object ...
        #     cv_fold         (instance) int64 ...
        #     label_nominal   (instance) object ...
        #     label_numeric   (instance) int64 ...
        #     arousal         (instance) float32 ...
        #     valence         (instance) float32 ...
        # Attributes:
        #     description:  features and labels of the IEMOCAP dataset

        if ".test" in name:

            if ".valid" in name:
                labels = numpy.array([d.decode('UTF-8') for d in data['label_nominal'].values])
            else:
                labels = data['label_nominal'].values

            indices_one_hot_angry = (labels == 'anger')
            angry_features = data['feature_value'].values[indices_one_hot_angry]
            angry_ids = data['file_name'].values[indices_one_hot_angry]

            indices_one_hot_happy1 = (labels == 'happiness')
            indices_one_hot_happy2 = (labels == 'excitement')
            indices_one_hot_happy = numpy.logical_or(indices_one_hot_happy1, indices_one_hot_happy2)
            happy_features = data['feature_value'].values[indices_one_hot_happy]
            happy_ids = data['file_name'].values[indices_one_hot_happy]

            indices_one_hot_sad = (labels == 'sadness')
            sad_features = data['feature_value'].values[indices_one_hot_sad]
            sad_ids = data['file_name'].values[indices_one_hot_sad]

            indices_one_hot_neutral = (labels == 'neutral')
            neutral_features = data['feature_value'].values[indices_one_hot_neutral]
            neutral_ids = data['file_name'].values[indices_one_hot_neutral]

            ids = numpy.concatenate([angry_ids, happy_ids, sad_ids, neutral_ids])
            features = numpy.concatenate([angry_features, happy_features, sad_features, neutral_features])
            labels = ['angry' for f in range(len(angry_features))] + ['happy' for f in range(len(happy_features))] + ['sad' for f in range(len(sad_features))] + ['neutral' for f in range(len(neutral_features))]

        else:
            labels = numpy.array([d.decode('UTF-8') for d in data['label_nominal'].values])
            indices = (labels == 'X')
            ids = data['file_name'].values[indices]
            features = data['feature_value'].values[indices]
            labels = ['X' for f in range(len(features))]

        print("extracted " + str(len(labels)) + " examples")
        print(features.shape)
        with open("data_" + name, 'wb') as f_out:
            pickle.dump(ids, f_out)
            pickle.dump(features, f_out)
            pickle.dump(labels, f_out)


loadData()

