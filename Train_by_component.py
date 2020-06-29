'''
Copied from unpack_and_split
Train separate ANN for each component

Unpacks data generated from ln_phi_data
Splits into X_train_full, X_test, y_train_full, y_test
Remove 'Phase' attrib. No categorical attribs remaining.
Commented out cat pipeline
'''

import os
import csv
import pandas as pd
import tarfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import sys
import tensorflow as tf
from tensorflow import keras
import time
import datetime
import subprocess
import pickle

# Read data
'''
with open('data_file.csv') as csvfile:
    lnphi = csv.reader(csvfile, delimiter=',')
    for row in lnphi:
        print(row)
        print(row[0])
        print(row[0],row[1],row[2],)
        break
'''

# Force CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Read using pandas. # CHANGE PATH FOR NEW ENV
LNPHI_PATH =r"C:\Users\win7\PycharmProjects\ML_PGE\ML Learn"
def load_lnphi_data(lnphi_path=LNPHI_PATH):
    csv_path = os.path.join(lnphi_path, "data_file.csv")
    return pd.read_csv(csv_path, delimiter=',', names=['Row_Index', 'P', 'T',#2
                                                       'x_N2', 'x_CO2', 'x_CH4', 'x_C2H6', 'x_C3H8', 'x_C4H10', 'x_C5H12', 'x_C6H14', 'x_PC-1', 'x_PC-2', 'x_PC-3', 'x_PC-4',#14
                                                       'lnphi_N2', 'lnphi_CO2', 'lnphi_CH4', 'lnphi_C2H6', 'lnphi_C3H8', 'lnphi_C4H10', 'lnphi_C5H12', 'lnphi_C6H14', 'lnphi_PC-1', 'lnphi_PC-2', 'lnphi_PC-3', 'ln_phi_PC-4',#26
                                                       'apure_N2', 'apure_CO2', 'apure_CH4', 'apure_C2H6', 'apure_C3H8', 'apure_C4H10', 'apure_C5H12', 'apure_C6H14', 'apure_PC-1', 'apure_PC-2', 'apure_PC-3', 'apure_PC-4',#38
                                                       'bpure_N2', 'bpure_CO2', 'bpure_CH4', 'bpure_C2H6', 'bpure_C3H8', 'bpure_C4H10', 'bpure_C5H12', 'bpure_C6H14', 'bpure_PC-1', 'bpure_PC-2', 'bpure_PC-3', 'bprue_PC-4',#50
                                                       'a_mix_phase (Am)', 'b_mix_phase (Bm)', 'Z_Root', 'b_beta', 'Phase'])#55

lnphi = load_lnphi_data()

# Drop 'Phase'
lnphi.drop(lnphi[lnphi['Phase']==1].index, inplace=True)
# Drop 'T'
lnphi.drop('T', axis=1, inplace=True)

# Define/Separate attributes and labels as X, y.
X = lnphi.iloc[:, 1:14].copy() # is .copy() needed?
#X['Phase'] = lnphi.iloc[:, 55] # Appends the 'Phase' column
y = lnphi.iloc[:, 14:26].copy()
del lnphi
#################################################################
# For component-specific model training, drop all attributes other than specific component lnphi
y = y['lnphi_N2'].copy()

# Split into train_full and test sets.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Further split train_full -> (train, valid) sets.
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42)

################################################################
# Define some classes for transformation pipeline. Classes will be called sequentially in a pipeline to
# transform data.

# Convert DataFrame to np array
from sklearn.base import BaseEstimator, TransformerMixin # BaseEstimate gets extra methods get_params and set_params
# TransformerMixin gets fit_transform()
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# Dummy transformer, just to learn.
class dummy_transformer(BaseEstimator, TransformerMixin):
    #def __init__(self):
        #nothing
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X

from sklearn.preprocessing import MinMaxScaler

# Combined pipeline. Cat pipeline omitted since there are no categories. Phase category is already OneHot number. See main_housing_exercise.
from sklearn.pipeline import Pipeline
#X_train_full_num = X_train_full.drop('Phase', axis=1)
X_train_full_num = X_train_full
num_attribs = list(X_train_full_num)
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('dummy_transform', dummy_transformer()),
    ('MinMaxScaler', MinMaxScaler()) # Does this limit generalizability?
    # Must remember training min and max for transforming new data.
])

cat_attribs = ['Phase'] # why not as a list?
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('dummy_transform', dummy_transformer())
])

from sklearn.pipeline import FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline)#,
    #("cat_pipeline", cat_pipeline),
])

X_train_prepared = full_pipeline.fit_transform(X_train) # We want to fit on training set, then transform on training and validation sets. (And finally on test set once its validated)
X_valid_prepared = full_pipeline.transform(X_valid)

##################################################################

n_inputs = X_train_prepared.shape[1]
model = keras.models.Sequential([
    keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_shape=[n_inputs]),
    keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    keras.layers.Dense(1)
])
# Remove lr if scheduler in use?
model.compile(loss='mse', optimizer=keras.optimizers.Adam(),
              metrics=['mse', 'mae'])
#####################################################################################
# Saving to file

# Logs callback
model_name = 'Test_lrelu_Adam_phase_0'
logdir = r"C:\Users\win7\Desktop"+".\\logs\\scalars\\" + model_name + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(logdir):
    os.makedirs(logdir)

tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=0,  # How often to log histogram visualizations
        write_graph=True,
        update_freq='epoch',
        profile_batch=0, # set to 0. Else bug Tensorboard not show train loss.
        embeddings_freq=0,  # How often to log embedding visualizations
    )

# Learning rate schedule as callback
def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.5 * (10 - epoch))
    '''if 0.001 * tf.math.exp(0.1 * (10 - epoch)) < 1E-5:
        return 1E-5
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))'''

lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

# Callback save
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=logdir,#+'.\\{epoch:.02d}-{mse:.2f}',
    verbose=1,
    save_weights_only=False,
    monitor='mse',# Not sure
    mode='auto',
    save_best_only=True)

# Store version info as file in directory
def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])
with open(logdir + '.\\version_info.txt', 'a', newline='\n') as file:
    file.write(str(get_git_revision_hash()) + '\n')

# Store attributes from data transformation
# Delete previous file
try:
    os.remove(logdir + '.\\full_pipeline_'+model_name+'.pkl')
except OSError:
    pass
with open(logdir + '.\\full_pipeline_'+model_name+'.pkl', 'wb') as f:
    pickle.dump(full_pipeline, f)

# "history" object holds a record of the loss values and metric values during training
history = model.fit(X_train_prepared, y_train, epochs=50, callbacks=[tensorboard_callback, lr_scheduler_callback, model_checkpoint_callback],
          validation_data=(X_valid_prepared, y_valid), shuffle=True, verbose=2)

model.save(logdir+'.\\'+model_name +'{}'.format(str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))))

endTime = datetime.datetime.now()
print('Ended at '+str(endTime))
print('end')
