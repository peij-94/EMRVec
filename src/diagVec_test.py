from .rae import RAE
import json
import os
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    MODELDIR = "../model"
    datetime = "181223_115609"
    options = json.load(
        open(os.path.join(MODELDIR, "rae_%s.model" % datetime, "rae_%s.params" % datetime), "r"))
    options["datetime"] = datetime
    model = RAE(options)

    data = pickle.load()
    label = pickle.load()

    diagVec_result = []
    label_result = []
    for i, _ in tqdm(data):
        _mask = np.zeros((len(_), options["max_seq_length"] - 1))
        diagVec_result.append(np.array(model.transform(_, _mask)).sum(axis=1))
        label_result.append(label[i, "isDied"])

    train_ind, test_ind = train_test_split(diagVec_result)

    rf = RFC()
    rf.fit(diagVec_result[train_ind], label_result[train_ind])
    label_pred = rf.predict(diagVec_result[test_ind])


