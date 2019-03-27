import os
MODELDIR = os.path.join("..", "model")
OUTDIR = os.path.join("..", "model")
PATIENTFILE = ""
DIGNOSESFILE = ""
ADMISSIONFILE = ""

options = {
    "num_inputs": 0,
    "num_hide1": 300,
    "num_hide2": 300,
    "num_hide3": 300,
    "epoch": 10,
    "batch_size": 1000,
    "shuffle": True
}