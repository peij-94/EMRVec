import pandas as pd
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

def apply_parelle(df_grouped, func, col=None, n_jobs=-1, types=1):
    if col:
        result = Parallel(n_jobs=n_jobs)(delayed(func)(df, col) for _, df in tqdm(df_grouped))
    else:
        result = Parallel(n_jobs=n_jobs)(delayed(func)(df) for _, df in tqdm(df_grouped))
    if types == 1:
        return pd.concat(result)
    else:
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-A", "--Admissions", dest="admissions",
                        help="the record of ADMISSIONS (patients' admissions info) in MIMIC-III",
                        required=True, type=str)
    parser.add_argument("-P", "--Patients", dest="patients", help="the record of PATIENTS (patients'info) in MIMIC-III",
                        required=True, type=str)
    parser.add_argument("-M", "--Medication", dest="medication",
                        help="the record of PRESCRIPTIONS (patients' medication record in MIMIC-III)", default="",
                        type=str)
    parser.add_argument("-D", "--Diagnoses", dest="diagnoses",
                        help="the record of DIAGNOSES_ICD (patients' diagnoses record with icd9 code in MIMIC-III)",
                        default="", type=str)
    parser.add_argument("-o", "--outdir", help="the output directory, default local dir", default=".", type=str)

    parser.add_argument("--adult", help="exclude newborn", action="store_true")

    args = parser.parse_args()

    assert len(args.diagnoses + args.medications) > 0, \
        "there must be more than one input in options (-P, -M, -D)"

    # processing admission info and patient info
    print("processing admission info and patient info ...")
    admission = pd.read_csv(args.admissions)
    patient = pd.read_csv(args.patients)
    patient["Btime"] = patient["DOB"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    admission["Atime"] = admission["ADMITTIME"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    patient["AGE"] = patient.apply(lambda x: int((admission.ix[
                                                      admission["SUBJECT_ID"] == x["SUBJECT_ID"], "Atime"].astype(
        datetime).values[0] - x["Btime"]).days / 365), axis=1)
    admission["AGE"] = admission.apply(lambda x: patient.ix[patient["SUBJECT_ID"] == x["SUBJECT_ID"], "AGE"].values[0],
                                       axis=1)
    # medication["STAYTIME"] = medication.apply(lambda x:(datetime.strptime(x["ENDDATE"], '%Y-%m-%d %H:%M:%S') - datetime.strptime(x["STARTDATE"], '%Y-%m-%d %H:%M:%S')).days + 1, axis=1)
    admission["ADMITTIME"] = admission["ADMITTIME"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    admission["DISCHTIME"] = admission["DISCHTIME"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    patient["AGE"] = patient["AGE"].apply(lambda x: 90 if x > 90 else x)

    if args.adult:
        ids = list(set(patient.ix[patient["AGE"]>0,"SUBJECT_ID"].tolist()))
        _prefix = "_adult"
    else:
        ids = list(set(patient["SUBJECT_ID"].tolist()))
        _prefix = ""

    if len(args.medication) > 0:
        print("processing medication ...")
        print("sequence extracted from the column 'DRUG' in PRESCRIPTIONS.csv ...")
        medication = pd.read_csv(args.medications)
        # processing miss value
        _ind = medication.ix[(medication["ENDDATE"].isnull()) & (medication["STARTDATE"].notnull())].index
        medication.ix[_ind, "ENDDATE"] = medication.ix[_ind, "STARTDATE"]
        _ind = medication.ix[(medication["STARTDATE"].isnull()) & (medication["ENDDATE"].notnull())].index
        medication.ix[_ind, "STARTDATE"] = medication.ix[_ind, "ENDDATE"]
        _ind = medication.ix[(medication["ENDDATE"].isnull()) & (medication["STARTDATE"].isnull())].index
        medication.drop(index=_ind, inplace=True)
        medication["STAYTIME"] = medication.apply(lambda x: (datetime.strptime(x["ENDDATE"],
                                                                               '%Y-%m-%d %H:%M:%S') - datetime.strptime(
            x["STARTDATE"], '%Y-%m-%d %H:%M:%S')).days + 1, axis=1)
        _ind = medication.ix[medication["STAYTIME"] < 1].index
        medication.ix[_ind, "ENDDATE"] = medication.ix[_ind, "STARTDATE"]
        medication.ix[_ind, "STAYTIME"] = 1
        # medication = medication.dropna(subset=["NDC"])
        # medication["NDC"] = medication["NDC"].apply(lambda x: "%011d" % (int(x)))
        # medication = medication.drop(index=medication.ix[medication["NDC"] == '00000000000'].index)
        medication["STARTDATE"] = medication["STARTDATE"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        medication["ENDDATE"] = medication["ENDDATE"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))


        def chk_data_error(df):
            return df.apply(lambda x: [
                1 if x["STARTDATE"] >= admission.ix[admission["HADM_ID"] == x["HADM_ID"], "ADMITTIME"].values[0] and x[
                    "ENDDATE"] <= admission.ix[admission["HADM_ID"] == x["HADM_ID"], "DISCHTIME"].values[0] else 0][0],
                            axis=1)


        mgroup = medication.groupby(by="HADM_ID")
        print("clearing some records with error date ...")
        medication["date_error"] = apply_parelle(mgroup, chk_data_error)
        medication.drop(index=medication.ix[medication["date_error"] == 0].index, inplace=True)


        def merge_mseq(df, col):
            #     df["STARTDATE"] = df["STARTDATE"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            #     df["ENDDATE"] = df["ENDDATE"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            df.sort_values(by=["STARTDATE", "ENDDATE"], inplace=True)
            start = df["STARTDATE"].values[0]
            end = df["ENDDATE"].sort_values(ascending=False).values[0]
            result = []
            for _ in range(int((end - start).astype("timedelta64[D]") / np.timedelta64(1, "D")) + 1):

                tmp = df.ix[(df["STARTDATE"] - start <= pd.Timedelta(_, "D")) & (
                            df["ENDDATE"] - start >= pd.Timedelta(_, "D")), col].tolist()
                if len(tmp) > 0:
                    result.append(tmp)
            # output record with subject_id
            return df["SUBJECT_ID"].values[0], result

        grouped = medication.ix[medication["SUBJECT_ID"].isin(ids)].groupby(by="SUBJECT_ID")
        seqs = apply_parelle(grouped, merge_mseq, "DRUG", types=2)
        types = []
        mseqs = []
        for i, _ in tqdm(seqs):
            mseq = []
            # filter some short-length seq
            if len(_) < 3:
                continue
            for item in _:
                s = []
                for it in item:
                    if it not in types:
                        types[it] = len(types) + 1
                    s.append(types[it])
                mseq.append(s)
            mseqs.append((i, seq))

        pickle.dump(mseqs, open("mseqs_name_by_patients%s.pkl" % (_prefix), "wb"))
        pickle.dump(types, open("mseqs_by_patients%s.types" % (_prefix), "wb"))

        print("finish...")

    if len(args.diagnoses) > 0:
        print("processing diagnoses ...")
        diagnoses = pd.read_csv(args.diagnoses)

        def func(df):
            result = []
            for i, _ in df.groupby("HADM_ID"):
                # sort icd9 code in seq by "SEQ_NUM"
                result.append(_.sort_values("SEQ_NUM", ascending=True)["ICD9_CODE"].astype(str).tolist())
            return df["SUBJECT_ID"].values[0], result

        grouped = diagnoses.ix[diagnoses["SUBJECT_ID"].isin(ids)].groupby(by="SUBJECT_ID")
        seqs = apply_parelle(grouped, func, types=2)
        types = {}
        dseqs = []
        for i, _ in tqdm(seqs):
            dseq = []
            # filter some short-length seq
            if len(_) < 3:
                continue
            for item in _:
                s = []
                for it in item:
                    if it not in types:
                        types[it] = len(types) + 1
                    s.append(types[it])
                dseq.append(s)
            dseqs.append((i, dseq))

        pickle.dump(dseqs, open("diagnoses_name_by_patients%s.pkl" % (_prefix), "wb"))
        pickle.dump(types, open("diagnoses_by_patients%s.types" % (_prefix), "wb"))

        print("finish...")
