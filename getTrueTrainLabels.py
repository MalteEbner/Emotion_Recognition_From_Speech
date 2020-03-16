import xarray as xr
import numpy
import pickle

def loadData():
    directory = "/Users/malteebner/Library/Mobile Documents/com~apple~CloudDocs/9. Semester/DL for NLP/Project/emotion_recognition/"  # exchange with the path to the data you have downloaded from Ilias

    names = ['prosody.test', 'logMel.test']
    file_paths = [directory + "data_prosody.test", directory + "data_logMel.test"]

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
            labels = ['angry' for f in range(len(angry_features))] + ['happy' for f in range(len(happy_features))] + ['sad' for f in range(len(sad_features))] + ['neutral' for f in range(len(neutral_features))]

            return ids, labels


ids, labels = loadData()
idLabelDict = dict(zip(ids, labels))
ypredFilename = "/Users/malteebner/Library/Mobile Documents/com~apple~CloudDocs/9. Semester/DL for NLP/Project/results_01_02_2019/3324181_ebner_topic2_result.txt"
correct = 0
lines = 0
with open(ypredFilename) as infile:
    for line in infile.readlines():
        lines += 1
        pred_id, pred_label = line.split(' ')
        if idLabelDict[pred_id] == pred_label:
            correct +=1
print float(correct)/lines



