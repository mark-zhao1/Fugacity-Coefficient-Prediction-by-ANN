import os
import pandas as pd
import tensorflow as tf
import subprocess
import pickle
import datetime
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer



class train_lnphi:
    def __init__(self):
        # Force CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return

    # Read using pandas
    def load_lnphi_data(self, lnphi_path, datafile_name):
        csv_path = os.path.join(lnphi_path, datafile_name) # previously data = "data_const_T_20200716-230921.csv"
        self.lnphi = pd.read_csv(csv_path, delimiter=',', names=['a_mix', 'b_mix', 'b_i', 'sum', 'lnphi'])
        print('Loading done. Shape: {}'.format(str(self.lnphi.shape)))

    # Drop out of range lnphi instances
    def lnphi_range(self, min, max):
        self.lnphi.drop(self.lnphi.loc[(self.lnphi.loc[:,'lnphi'] < min) | (self.lnphi.loc[:,'lnphi'] > max)].index,
                        inplace=True)
        print('Drop lnphi out of range done. Shape: {}'.format(str(self.lnphi.shape)))



    def split_data(self):
        self.X = self.lnphi.loc[:,'a_mix':'sum']
        self.y = self.lnphi.loc[:,'lnphi']

        # Split data -> (train_full, test)
        self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(
            self.X, self.y, test_size=0.1, random_state=42)

        # Split train_full -> (train, valid)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            self.X_train_full, self.y_train_full, test_size=0.1, random_state=42)

        print('Splitting done.')

    def feature_eng(self):
        # Label Transform pipeline
        self.label_scaler = MinMaxScaler()
        self.label_num_pipeline = Pipeline([
            ('label minmax scaler', self.label_scaler)
        ])
        self.y_train_prepared = self.label_num_pipeline.fit_transform(self.y_train.values.reshape(-1,1))
        self.y_valid_prepared = self.label_num_pipeline.transform(self.y_valid.values.reshape(-1,1))
        self.y_test_prepared = self.label_num_pipeline.transform(self.y_test.values.reshape(-1,1))

        # Attribute Transform pipeline

        self.attr_scaler = MinMaxScaler()
        num_pipeline = Pipeline([
            #('std scaler', self.attr_std_scaler)
            ('min_max_scaler', self.attr_scaler)
        ])
        num_attribs = list(self.X_train)
        self.full_pipeline = ColumnTransformer([
            ('num', num_pipeline, num_attribs)
        ])

        self.X_train_prepared = self.full_pipeline.fit_transform(self.X_train)
        self.X_valid_prepared = self.full_pipeline.transform(self.X_valid)
        self.X_test = self.full_pipeline.transform(self.X_test)

        print('Feature Eng done.')

    def model_construct(self, n_layers, n_nodes):
        n_inputs = self.X_train_prepared.shape[1]

        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.Dense(n_nodes, activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_shape=[n_inputs]))
        for _ in range(n_layers-1):
            self.model.add(tf.keras.layers.Dense(n_nodes, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
        self.model.add(tf.keras.layers.Dense(1))

        # Remove lr if scheduler in use?
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(),
                      metrics=['mse', 'mae', tf.keras.metrics.MeanAbsolutePercentageError()])

    def train_model(self, batch_size, n_layers, n_nodes, epochs, initial_epoch, log_save_dir, name_prefix):
        # Logs callback
        model_name = name_prefix+'_'+str(batch_size)+'_'+str(n_layers)+'_'+str(n_nodes)+'_'+str(epochs)+'_'
        try:
            logdir = self.logdir
        except AttributeError:
            print('New logdir created.')
            self.logdir = log_save_dir + ".\\logs\\scalars\\" + model_name + str(
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            logdir = self.logdir
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=0,  # How often to log histogram visualizations
            write_graph=True,
            update_freq='epoch',
            profile_batch=0,  # set to 0. Else bug Tensorboard not show train loss.
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

        #lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

        #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=0.0001)

        # Early stop
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='mse', min_delta=0.001, patience=3)
        #todo maybe make proportional early stopping

        # Callback save
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=logdir,  # +'.\\{epoch:.02d}-{mse:.2f}',
            verbose=1,
            save_weights_only=False,
            monitor='mse',  # Not sure
            mode='auto',
            save_best_only=True)

        # Store version info as file in directory
        def get_git_revision_hash():
            return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

        with open(logdir + '.\\version_info.txt', 'a', newline='') as file:
            file.write('model_name'+' '+str(get_git_revision_hash()) + '\n')

        # Store attributes from data transformation
        # Delete previous file if exists
        try:
            os.remove(logdir + '.\\full_pipeline_' + model_name + '.pkl')
        except OSError:
            pass
        with open(logdir + '.\\full_pipeline_' + model_name + '.pkl', 'wb') as f:
            pickle.dump(self.full_pipeline, f)
            pickle.dump(self.label_num_pipeline, f)

        # "history" object holds a record of the loss values and metric values during training
        history = self.model.fit(self.X_train_prepared, self.y_train_prepared, initial_epoch=initial_epoch, epochs=epochs,
                            callbacks=[tensorboard_callback, model_checkpoint_callback],
                            validation_data=(self.X_valid_prepared, self.y_valid_prepared), shuffle=True,
                            batch_size=batch_size, verbose=2)

        # Save entire model with training config
        self.model.save(logdir + '.\\' + model_name + '{}'.format(str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))))

        endTime = datetime.datetime.now()
        print('Ended at ' + str(endTime))
        print('end')


if __name__=='__main__':
    LNPHI_PATH = r"E:\Datasets"
    log_save_dir = r"C:\Users\win7\Desktop\logs"
    tr = train_lnphi()
    tr.load_lnphi_data(LNPHI_PATH)
    tr.lnphi_range(-50, 50)
    tr.split_data()
    tr.feature_eng()

    # Define some hyperparameters
    list_batch_size = [512, 100] #1024,
    list_n_layers = [2, 8, 32]
    list_n_nodes =[20, 40]
    epochs = 30 # max epochs if use early stopping
    for batch_size in list_batch_size:
        for n_layers in list_n_layers:
            for n_nodes in list_n_nodes:
                if batch_size == 512 and n_layers==2 and n_nodes==20:
                    continue
                print('Training with batch_size: {}, n_layers: {}, n_nodes: {}.'.format(batch_size, n_layers, n_nodes))
                tr.model_construct(n_layers, n_nodes)
                tr.train_model(batch_size, n_layers, n_nodes, epochs, log_save_dir)
    print('end')
    #todo check distribution of lnphi via histogram? Maybe some transformation on it is required? RMSE affected by target distribution