import pandas as pd
from datetime import datetime
from tqdm import tqdm
import math
from config import *
from autoencoder import *
import numpy as np

if __name__ == "__main__":

    # data-processing
    print("preparing data ...")
    patient = pd.read_csv(PATIENTFILE)
    diagnoses = pd.read_csv(DIGNOSESFILE)
    admission = pd.read_csv(ADMISSIONFILE)

    patient["Btime"] = patient["DOB"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    admission["Atime"] = admission["ADMITTIME"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    patient["AGE"] = patient.apply(lambda x: int((admission.ix[admission["SUBJECT_ID"] == x["SUBJECT_ID"], "Atime"].astype(datetime).values[0] - x["Btime"]).days / 365), axis=1)
    diagnoses.dropna(inplace=True)
    patient["AGE"] = patient["AGE"].apply(lambda x: 90 if x > 90 else x)

    died_in_hosp = patient.ix[patient["SUBJECT_ID"].isin(admission.ix[admission["DEATHTIME"].notnull(), "SUBJECT_ID"].tolist())]
    died_in_hosp["DISCHTIME"] = died_in_hosp.apply(lambda x: sorted(admission.ix[admission["SUBJECT_ID"] == x["SUBJECT_ID"], "DISCHTIME"])[-1], axis=1)

    data = died_in_hosp

    # build-model
    print("building model ...")
    loss, optimizer = build_model(options)

    # train-model
    print("training model ...")
    for i in range(options["epoch"]):
        if options["shuffle"]:
            np.random.shuffle(data.index.values)
        batch_num = int(math.floor(data.shape[0]/options["batch_size"]))
        for batch in range(batch_num):
            if batch == batch_num - 1:
                data_batch = data.ix[data.index.values[batch*options["batch_size"]:]]
            else:
                data_batch = data.ix[data.index.values[batch*options["batch_size"]:(batch_num+1)*options["batch_size"]]]
            


