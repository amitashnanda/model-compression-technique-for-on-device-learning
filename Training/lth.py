import tempfile
import os
import argparse
import csv 
import numpy as np
import math
import random
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import regularizers

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from utils import stats_report, save_model, loaf_tf_lite_model
from cosine_annealing import CosineAnnealingScheduler
from swa.tfkeras import SWA

# Define dataset
class IEGM_DataSET():
    def __init__(self, root_dir, indice_dir, mode, size, transform=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = self.root_dir + self.names_list[idx].split(' ')[0]

        if not os.path.isfile(text_path):
            print(text_path + 'does not exist')
            return None

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'IEGM_seg': IEGM_seg, 'label': label}

        return sample

def loadCSV(csvf):
    """
    return a dict saving the information of csv
    :param splitFile: csv file name
    :return: {label:[file1, file2 ...]}
    """
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels

def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, root_dir, indice_dir, mode, size):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))

  def __len__(self):
    return len(self.names_list)

  def __getitem__(self, idx):
    text_path = self.root_dir + self.names_list[idx].split(' ')[0]
    if not os.path.isfile(text_path):
      print(text_path + 'does not exist')
      return None

    IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
    label = int(self.names_list[idx].split(' ')[1])
    # sample = np.array(IEGM_seg, label)
    sample = np.append(IEGM_seg, label)
    # sample = {'IEGM_seg': IEGM_seg, 'label': label}
    return sample

# Define the model architecture.

def model_best():
  model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(1250, 1)),
      tf.keras.layers.Conv1D(filters=3, kernel_size=85, strides=32, padding='valid', activation=None, use_bias=True),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Flatten(),

      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(20),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(10),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(2),
  ])
  return model


def step_decay(step):
  initial_learning_rate = 0.0004
  decay_steps = 100
  alpha = 0.0001
  step = min(step, decay_steps)
  cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
  decayed = (1 - alpha) * cosine_decay + alpha
  return initial_learning_rate * decayed

import tensorflow as tf
from tensorflow.keras import layers, models

def lth_pruning(pruning_percent, target_ratio):
    # Step 2: Randomly initialize the given DL network
    model, x_aug, y_aug, args, model_checkpoint_callback, x_test, y_test = get_model()

    while True:
        # Step 3: Train the DL network with the given data
        model.compile(optimizer=Adam(lr=args.lr),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
        )  # You may need to adjust the optimizer and loss based on your task
        
        model.fit(
                x_aug,
                y_aug,
                epochs=2,
                batch_size=args.batchsz,
                validation_split=0.3,
                shuffle=True,
                # validation_data=(x_test, y_test),
                callbacks=[model_checkpoint_callback]
            )
        # Step 4: Prune pruning_percent% of weights with least magnitude
        model = prune_model(model, pruning_percent)

        # Step 5: Check the remaining ratio of weights
        current_ratio = calculate_nonzero_params_ratio(model)
        print(f"Remaining ratio: {current_ratio}")
        
        # Step 5: if current_ratio >= target_ratio, STOP and output the pruned model
        if current_ratio >= target_ratio:
            print("Pruning complete.")

            pred = model.predict(x_test).argmax(axis=1)
            segs_TP = 0
            segs_TN = 0
            segs_FP = 0
            segs_FN = 0

            for predicted_test, labels_test in zip(pred, y_test.numpy()):
                if labels_test == 0:
                    segs_FP += (1 - (predicted_test == labels_test).sum()).item()
                    segs_TN += (predicted_test == labels_test).sum().item()
                elif labels_test == 1:
                    segs_FN += (1 - (predicted_test == labels_test).sum()).item()
                    segs_TP += (predicted_test == labels_test).sum().item()
            FB = stats_report([segs_TP, segs_FN, segs_FP, segs_TN])

            return FB, model

        # Step 6: Adjust pruning_percent
        pruning_percent *= 1.1

        # Step 7: Reset the model
        model.set_weights(model.get_weights())

def prune_model(model, pruning_percent):
    # Flatten and sort weights
    weights = []
    for layer in model.layers:
        if isinstance(layer, layers.Conv1D) or isinstance(layer, layers.Dense):
            weights.extend(tf.reshape(layer.get_weights()[0], [-1]))
    weights = tf.sort(tf.abs(weights))

    # Calculate threshold for pruning
    threshold_idx = int(pruning_percent * len(weights))
    threshold = weights[threshold_idx]

    # Prune weights below the threshold
    for layer in model.layers:
        if isinstance(layer, layers.Conv1D) or isinstance(layer, layers.Dense):
            weights = layer.get_weights()
            weights[0][tf.abs(weights[0]) < threshold] = 0
            layer.set_weights(weights)

    return model

def calculate_nonzero_params_ratio(model):
    total_params = 0
    nonzero_params = 0

    for layer in model.layers:
        if isinstance(layer, layers.Conv1D) or isinstance(layer, layers.Dense):
            total_params += tf.size(layer.get_weights()[0])
            nonzero_params += tf.math.count_nonzero(layer.get_weights()[0])

    return float(nonzero_params) / float(total_params)

# Example usage:
# Replace 'your_ecg_data' with your actual training ECG data
# Replace 'your_dl_network' with your actual deep learning network
# Set the desired target_ratio
# lth_pruning(your_ecg_data, your_dl_network, pruning_percent=30, target_ratio=0.2)

def get_model():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0002)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=4)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./tinyml_contest_data_training/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')
    
    args = argparser.parse_args()

    # Hyperparameters
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    EPOCH =  20 #args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    # Data aug setting
    data_aug = True
    mix = False
    flip_peak = True
    flip_time = False
    add_noise = True

    train_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode='train', size=SIZE)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_generator)
    train_dataset = train_dataset.shuffle(10).batch(len(train_generator))
    train_dataset = train_dataset.repeat()
    train_iterator = iter(train_dataset)

    test_generator = DataGenerator(root_dir=path_data, indice_dir=path_indices, mode='test', size=SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_generator)
    test_dataset = test_dataset.shuffle(10).batch(len(test_generator))
    test_dataset = test_dataset.repeat()
    test_iterator = iter(test_dataset)

    one_element = train_iterator.get_next()
    x, y = one_element[...,0:-1], one_element[...,-1]
    x = np.expand_dims(x, axis=2)

    test_samples = test_iterator.get_next()
    x_test, y_test = test_samples[...,0:-1], test_samples[...,-1]
    x_test = np.expand_dims(x_test, axis=2)
    
    if data_aug:
      if mix:
        x_aug, y_aug = np.concatenate((x, x_test), axis=0), np.concatenate((y, y_test), axis=0)
        print('Mix Data Finish!')    
      else:
        x_aug = np.copy(x)
        y_aug = np.copy(y)
        for i in range(len(x)):
          flip_p = random.random()
          flip_t = random.random()
          if flip_p < 0.5 and flip_peak:
            x_aug[i] = -x[i]
          if flip_t < 0.5 and flip_time:
            x_aug[i] = np.flip(x[i])
          if add_noise:
            max_peak = x_aug[i].max() * 0.05
            factor = random.random()
            # factor = 1
            noise = np.random.normal(0, factor * max_peak, (len(x_aug[i]), 1))
            x_aug[i] = x_aug[i] + noise

        print('flip Peak: ', flip_peak)
        print('Add Noise: ', add_noise) 
    
    start_epoch = 10
    swa = SWA(start_epoch=start_epoch, 
          lr_schedule='cyclic', 
          swa_lr=0.0001,
          swa_lr2=0.0005,
          swa_freq=5,
          batch_size=args.batchsz,
          verbose=1)

    my_model = model_best()
    save_name = 'random_' + 'lth'
    # save_name = 'SWA' 
    checkpoint_filepath = './20_10/' + save_name + '/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Train the digit classification model
    # lrate = LearningRateScheduler(step_decay)
    lrate = CosineAnnealingScheduler(T_max=100, eta_max=4e-4, eta_min=2e-4)

    return my_model, x_aug, y_aug, args, model_checkpoint_callback, x_test, y_test
    
if __name__ == '__main__':
    best_FB = 0.
    FB, model = lth_pruning(0.3, .2)
    save_model(f"best_model.tflite", model)
    print(FB)
    # for i in range(10):
    #     FB, my_model = run_once(i)
    #     if FB > best_FB:
    #         best_FB = FB
    #         save_model(f"best_model.tflite", my_model)
    #         print('Current Best: ', best_FB)
    #     print(FB)
    # print('Current Best: ', best_FB)