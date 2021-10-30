import glob
import math
import ntpath
import random, os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import edfreader
import models
import BNN_model
import mne
# import pyeeg
import warnings

warnings.filterwarnings('ignore')

from matplotlib.backends.backend_pdf import PdfPages
from mne.datasets.sleep_physionet._utils import _fetch_one, _data_path, AGE_SLEEP_RECORDS, _check_subjects
from datetime import datetime
from mne import Epochs, pick_types, find_events
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import psd_welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import FunctionTransformer
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
# from tensorflow import keras
# from tensorflow.keras import optimizers, losses
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, MaxPool1D, Activation, SpatialDropout1D, GlobalMaxPool1D
# from tensorflow.keras.layers import Reshape, LSTM, TimeDistributed, Bidirectional, BatchNormalization, Flatten, RepeatVector
# from tensorflow.keras.layers import concatenate
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
# from sklearn.externals import joblib
# from sklearn.preprocessing import StandardScaler
# from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, classification_report, log_loss
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_auc_score, roc_curve
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
import xgboost


# Library Imports --------------------------------
# from fetch_data import fetch_data

VBS = True  # constant boolean to enable/disbale verbose
EPOCH_SEC_SIZE = 30  # Epoch duration selection
seed = 42  # seed value for the random seeds
batch_size = 64
number_of_subj = 1



# load files:
folder = "physionet-sleep-data" 
filenames = os.listdir(folder)
filenames.sort()
index = 0
data_temp = []
final_list = []
for file in filenames:
    if index%3 == 0:
        data_temp.append('physionet-sleep-data/'+file)
    if index%3 == 1:
        data_temp.append('physionet-sleep-data/'+file)
        final_list.append(data_temp)
        data_temp = []
    index += 1

        



# values to label the stages
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
subject_files = final_list
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
# PSD = []  # Power Spectral Density
# PFD = []  # Petrosian Fractal Dimension
# hjorths = []  # Hjorth Parameters
# hursts = []  # Hurst Exponent
# DFA = []  # Detrended Fluctuation Analysis
# for item in tqdm(subject_files[:5]):
#     raw_test = mne.io.read_raw_edf(item[0], verbose=False)
#     signals_list = raw_test[0][0][0]
#     first_order = np.diff(signals_list).tolist()
#     PSD.append(pyeeg.bin_power(signals_list, band_list, Fs))
#     PFD.append(pyeeg.pfd(signals_list, first_order))
#     hjorths.append(pyeeg.hjorth(signals_list, first_order))
#     hursts.append(compute_Hc(signals_list, kind='change', min_window=256))
#     DFA.append(pyeeg.dfa(signals_list))


# if VBS:
#     print("Petrosian Fractal Dimension (PFD): ", PFD)
#     print("Hjorth mobility and complexity: ", hjorths)
#     print("Detrended Fluctuation Analysis (DFA): ", DFA)
#     print("Hurst Exponent (Hurst): ", hursts)
# for item in tqdm(subject_files):
#     filename = ntpath.basename(item[0]).replace("-PSG.edf", ".npz")  # reading the PSG files
#     if not os.path.exists(os.path.join(output_path, filename)):
#         raw_train = mne.io.read_raw_edf(item[0], verbose=VBS)
#         sampling_rate = raw_train.info['sfreq']
#         raw_ch_df = raw_train.to_data_frame()[ch_labels]
#         # raw_ch_df = raw_ch_df.to_frame()
#         raw_ch_df.set_index(np.arange(len(raw_ch_df)))
        
#         # reading the raw headers using the EDFReader function from edfreader
#         f = open(item[0], 'r', errors='ignore', encoding='utf-8')
#         head_raw_read = edfreader.BaseEDFReader(f)
#         head_raw_read.read_header()
#         head_raw = head_raw_read.header
#         f.close()
#         raw_start_time = datetime.strptime(head_raw['date_time'], "%Y-%m-%d %H:%M:%S")

#         # read annotations from hypnogram file
#         f = open(item[1], 'r')
#         annot_raw_read = edfreader.BaseEDFReader(f)
#         annot_raw_read.read_header()
#         annot_raw = annot_raw_read.header
#         temp, temp, total_annot = zip(*annot_raw_read.records())
#         f.close()
#         annot_start_time = datetime.strptime(annot_raw['date_time'], "%Y-%m-%d %H:%M:%S")
#         assert raw_start_time == annot_start_time  # making sure that the PSG files and hypnogram files are in sync
#         remove_idx = []    # list to keep the indicies of data that will be removed
#         labels = []        # list to keep the indicies of data that have labels
#         label_idx = []
        
#         # selecting the indicies of known labels and adding the rest to remove_idx list
#         for annot in total_annot[0]:
#             onset_sec, duration_sec, annot_char = annot
#             annot_str = "".join(annot_char)
#             if (annot_str in annot2label.keys()) == False:
#                 label = -1
#             else:
#                 label = annot2label[annot_str]
#             if label != UNKNOWN:
#                 if duration_sec % EPOCH_SEC_SIZE != 0:
#                     raise Exception("Please choose anothe epoch duration!")
#                 duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
#                 label_epoch = np.ones(duration_epoch, dtype=np.int) * label
#                 labels.append(label_epoch)
#                 idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int)
#                 label_idx.append(idx)
#             else:
#                 idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int)
#                 remove_idx.append(idx)
#         labels = np.hstack(labels)
#         if len(remove_idx) > 0:
#             remove_idx = np.hstack(remove_idx)
#             select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
#         else:
#             select_idx = np.arange(len(raw_ch_df))

#         # filtering data with labels only
#         label_idx = np.hstack(label_idx)
#         select_idx = np.intersect1d(select_idx, label_idx)

#         # removing extra indicies
#         if len(label_idx) > len(select_idx):
#             extra_idx = np.setdiff1d(label_idx, select_idx)
#             # trimming the tail
#             if np.all(extra_idx > select_idx[-1]):
#                 n_trims = len(select_idx) % int(EPOCH_SEC_SIZE * sampling_rate)
#                 n_label_trims = int(math.ceil(n_trims / (EPOCH_SEC_SIZE * sampling_rate)))
#                 select_idx = select_idx[:-n_trims]
#                 labels = labels[:-n_label_trims]

#         # removing all unknown and movement labels
#         raw_ch = raw_ch_df.values[select_idx]

#         # check if we can split into epochs' size
#         if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
#             raise Exception("Please choose anothe epoch duration!")
#         n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

#         # get epochs and their corresponding labels
#         x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
#         y = labels.astype(np.int32)

#         assert len(x) == len(y)

#         # select on sleep periods
#         w_edge_mins = 30
#         nw_idx = np.where(y != label_dict["W"])[0]
#         start_idx = nw_idx[0] - (w_edge_mins * 2)
#         end_idx = nw_idx[-1] + (w_edge_mins * 2)
#         if start_idx < 0: start_idx = 0
#         if end_idx >= len(y): end_idx = len(y) - 1
#         select_idx = np.arange(start_idx, end_idx+1)
#         x = x[select_idx]
#         y = y[select_idx]

#         # file structure for saving
#         save_dict = {
#             "x": x, 
#             "y": y, 
#             "fs": sampling_rate,
            
#             "header_raw": head_raw,
#             "header_annotation": annot_raw,
#         }
#         # "ch_label": ch_labels,
#         if not os.path.exists(output_path):
#             os.makedirs(output_path)
#         np.savez(os.path.join(output_path, filename), **save_dict)

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
if VBS:
    print("Shape of the training dataset:\ntraining dataset: {}\ntest_dataset: {}\n"
          .format(X_train.shape, X_test.shape))
y_train_ = to_categorical(y_train)
y_val_ = to_categorical(y_val)
y_test_ = to_categorical(y_test)

# X_train = np.squeeze(X_train)
# X_test = np.squeeze(X_test)
# X_val = np.squeeze(X_val)

# pp_X_train = np.array([models.butter_bandpass_filter(sample, highpass=40.0, fs=256, order=4) for sample in X_train])
# pp_X_val = np.array([models.butter_bandpass_filter(sample, highpass=40.0, fs=256, order=4) for sample in X_val])
# pp_X_test = np.array([models.butter_bandpass_filter(sample, highpass=40.0, fs=256, order=4) for sample in X_test])
pp_X_train = np.array([sample for sample in X_train])
pp_X_val = np.array([sample for sample in X_val])
pp_X_test = np.array([sample for sample in X_test])

# pp_X_test = np.expand_dims(pp_X_test, axis=2)
# pp_X_train = np.expand_dims(pp_X_train, axis=2)
# pp_X_val = np.expand_dims(pp_X_val, axis=2)
if VBS:
    print(pp_X_val.shape)
    print(pp_X_train.shape)

# checkpoint = ModelCheckpoint("model_cps", monitor='val_loss', verbose=1, save_best_only=True, mode='max')
# redonplat = ReduceLROnPlateau(monitor="val_loss", mode="max", patience=5, verbose=2)
# csv_logger = CSVLogger('log_training.csv', append=True, separator=',')
# callbacks_list = [

#     redonplat

# ]

# model_cnn = BNN_model.create_bnn_model(verbose=VBS)
# train_dataset = tf.data.Dataset.from_tensor_slices((pp_X_train,y_train_)).batch(batch_size)
# val_dataset = tf.data.Dataset.from_tensor_slices((pp_X_val,y_val_)).batch(batch_size)


from torch_BNN import Passthrough
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
model_cnn = Passthrough()
import torch
import torch.nn as nn
import torchbnn as bnn
tensor_x = torch.Tensor(pp_X_train.reshape([pp_X_train.shape[0], 8, 7680])) # transform to torch tensor
tensor_y = torch.Tensor(y_train.reshape([y_train.shape[0], 1]).tolist())

my_dataset = TensorDataset(tensor_x.to("cuda"),tensor_y.to("cuda"))
my_dataloader = DataLoader(my_dataset, batch_size= batch_size)
# # pytroch version:





from torch_BNN import model_cnn, model_bnn, train_bnn, train_cnn, Passthrough

model = model_cnn().to("cuda")
# model = model_bnn().to("cuda")
# model = Passthrough()
# ce_loss = nn.NLLLoss()
# kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
optimizer = optim.Adam(model.parameters(), lr=0.0003)
kl_weight = 0.1

train_bnn(model, optimizer, my_dataloader, 64, 10)



pass






# hist_19 = model_cnn.fit(
#     train_dataset, epochs=30, validation_data=val_dataset, verbose=VBS
# )

# y_pred = model_cnn.predict(pp_X_test, batch_size=batch_size)
# y_pred = np.array([np.argmax(s) for s in y_pred])
# f1_cnn = f1_score(y_test, y_pred, average="macro")
# if VBS:
#     print("F1 score: {}".format(f1_cnn))
#     report = classification_report(y_test, y_pred)
#     print(report)

# plt.plot(hist_19.history["loss"])
# plt.plot(hist_19.history["val_loss"])

# plt.plot(hist_19.history["acc"])
# plt.plot(hist_19.history["val_acc"])