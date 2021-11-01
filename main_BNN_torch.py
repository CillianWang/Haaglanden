# import pyeeg
import warnings
import glob
import math
import ntpath
import random, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

from matplotlib.backends.backend_pdf import PdfPages
# from mne.datasets.sleep_physionet._utils import _fetch_one, _data_path, AGE_SLEEP_RECORDS, _check_subjects
from datetime import datetime
# from mne import Epochs, pick_types, find_events
# from mne.io import concatenate_raws, read_raw_edf
# from mne.time_frequency import psd_welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import FunctionTransformer
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import make_scorer, f1_score, accuracy_score, classification_report, log_loss
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_auc_score, roc_curve
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from xgboost import XGBClassifier
# import xgboost

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]
# Library Imports --------------------------------
# from fetch_data import fetch_data

VBS = True  # constant boolean to enable/disbale verbose
EPOCH_SEC_SIZE = 30  # Epoch duration selection
seed = 42  # seed value for the random seeds
batch_size = 64
number_of_subj = 3
lr = 0.0003
epoch_num = 200
class_num = 0
percentage = 0
gamma = 0.99

UNKNOWN = -1
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4

# making string dictionary for the label values
label_dict = {
    "UNKNOWN"  : UNKNOWN,
    "W"        : W,
    "N1"       : N1,
    "N2"       : N2,
    "N3"       : N3,
    "REM"      : REM
}

# converting from label values to strings 
class_dict = {
    -1: "UNKNOWN",
    0 : "W",
    1 : "N1",
    2 : "N2",
    3 : "N3",
    4 : "REM"
}

# annotation dictionary to convert from string to label values
annot2label = {
    "Sleep stage ?": -1,
    "Movement time": -1,
    "Lights off@@EEG F4-A1":-1,
    "Lights on@@EEG Fpz-Cz":-1,
    "Lights off@@EMG RA": -1,
    "Lights off@@EMG RAT": -1,
    "Lights on@@Resp chest": -1,
    "Lights off@@Resp chest": -1,
    "Lights off@@SaO2": -1, 
    "Lights on@@SaO2" : -1,
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage N4": 3,
    "Sleep stage R": 4
}
project_path = os.path.abspath(os.getcwd())  # finding the current project path in windows

# from mne.datasets.sleep_physionet._utils import _fetch_one, _data_path, AGE_SLEEP_RECORDS, _check_subjects
import numpy as np



subjects_list = []  # list to keep the address of the subject data
# except_sub = [13, 36, 52]  # omitting the subjects with incomplete data 
except_sub=[]
for i in range(83):
    if i in except_sub:
        continue
    subjects_list.append(i)
# fetching data of each subject and 
# subject_files = fetch_data(subjects=subjects_list, recording=[1, 2], path= project_path)  
# subject_files = [['physionet-sleep-data/SN001.edf', 'physionet-sleep-data/SN001_sleepscoring.edf']]
# subject_files = final_list
mapping = {'EEG O2-M1': 'eeg',
           'EEG F4-M1': 'eeg',
           'EEG C4-M1': 'eeg',
           'EEG C3-M2': 'eeg',
           'EOG E1-M2': 'eog',
           'EOG E2-M2': 'eog',
           'ECG': 'ecg',
           'EMG chin': 'emg'}

ch_labels = ['EEG F4-M1',  'EEG C4-M1',  'EEG O2-M1',  'EEG C3-M2',  'EMG chin',  'EOG E1-M2',  'EOG E2-M2',   'ECG']

# ch_labels = 'EEG F4-M1'  # channels to be selected
# ch_labels used when only one channel is selected!
data_frames = []
if VBS:
    print("Importing data into dataframes:")
output_path = os.path.join(project_path, "NPZ_files")  # path to save the npz files

Fs= 256
band_list = [0.5,4,7,12,30]

npz_files = sorted(glob.glob(os.path.join(output_path, "*.npz")))
X = np.zeros((0, 7680, 8))
y = []
for fn in tqdm(npz_files[:number_of_subj]):
    samples = np.load(fn)
    X_data = samples['x']
    X = np.concatenate((X, X_data), axis=0)
    y.extend(samples['y'])
y = np.array(y)

pd.Series(y).value_counts().plot.bar()
plt.title("Frequency of the labels in our dataset")

if VBS:
    print("Shape of the input data: {}".format(X.shape))
    print("Shape of the sleep stages: {}".format(y.shape))
# splitting subjects
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
# splitting sleeping signals
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)

def block_class(class_num, p, x, y):
    shape = y.shape[0]
    index = []
    for i in range(shape):
        if y[i] == class_num:
            dice = random.random()
            if dice < p:
                index.append(i)
    

    x = np.delete(x, index, 0)
    y = np.delete(y, index, 0)
    return x, y

X_train, y_train = block_class(class_num, percentage, X_train, y_train)


if VBS:
    print("Shape of the training dataset:\ntraining dataset: {}\ntest_dataset: {}\n"
          .format(X_train.shape, X_test.shape))
y_train_ = to_categorical(y_train, 5)
y_val_ = to_categorical(y_val, 5)
y_test_ = to_categorical(y_test, 5)

# pp_X_train = np.array([models.butter_bandpass_filter(sample, highpass=40.0, fs=256, order=4) for sample in X_train])
# pp_X_val = np.array([models.butter_bandpass_filter(sample, highpass=40.0, fs=256, order=4) for sample in X_val])
# pp_X_test = np.array([models.butter_bandpass_filter(sample, highpass=40.0, fs=256, order=4) for sample in X_test])
pp_X_train = np.array([sample for sample in X_train])
pp_X_val = np.array([sample for sample in X_val])
pp_X_test = np.array([sample for sample in X_test])

if VBS:
    print(pp_X_val.shape)
    print(pp_X_train.shape)


from torch_BNN import Passthrough
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
# model_cnn = Passthrough()
import torch
import torch.nn as nn
import torchbnn as bnn



tensor_x = torch.Tensor(pp_X_train.reshape([pp_X_train.shape[0], 8, 7680])) # transform to torch tensor
tensor_y = torch.Tensor(y_train.reshape([y_train.shape[0], 1]).tolist())
tensor_x_val = torch.Tensor(pp_X_val.reshape([pp_X_val.shape[0], 8, 7680]))
tensor_y_val = torch.Tensor(y_val.reshape([y_val.shape[0], 1]).tolist())


my_dataset = TensorDataset(tensor_x.to("cuda"),tensor_y.to("cuda"))
my_dataloader = DataLoader(my_dataset, batch_size= batch_size)
val_dataset = TensorDataset(tensor_x_val.to("cuda"),tensor_y_val.to("cuda"))
val_dataloader = DataLoader(val_dataset, batch_size = batch_size)

# # pytroch version:






from torch_BNN import model_cnn, model_bnn, train_bnn, train_cnn, Passthrough

model = model_cnn().to("cuda")
#model = model_bnn().to("cuda")
# model = Passthrough()
# ce_loss = nn.NLLLoss()
# kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
optimizer = optim.Adam(model.parameters(), lr=lr)
kl_weight = 0.1

train_bnn(model, optimizer, my_dataloader, val_dataloader=val_dataloader, batch_size=batch_size, epochs=epoch_num, gamma = gamma)

train_cnn(model, optimizer, my_dataloader, val_dataloader=val_dataloader, batch_size=batch_size, epochs=epoch_num)

pass

