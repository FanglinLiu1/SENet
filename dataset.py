# ------------------------------------------------------------------------------------------------
# Integrated AutoEncoder
# Copyright (c) 2024 Fanglin Liu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------

import pickle
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['DatasetLoader']


class DatasetLoader:
    def __init__(self, dataset_path="./dataset/RML2016.10a_dict.pkl"):
        self.dataset_path = dataset_path
        self.Xd = None
        self.mods = None
        self.snrs = None
        self.X = None
        self.lbl = None
        self.mod_snr_data = None
        self.mod_count = None
        self.snr_count = None

    def preprocess_data(self, X, mod_snr_data):
        # Standardization
        # mean = X.mean(axis=0)
        # std = X.std(axis=0)
        # std[std == 0] = 1
        # X = (X - mean) / std

        # Normalization: [-1, 1]
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1
        X = 2 * (X - X_min) / X_range - 1

        # X = {}
        # for mod_snr, data in mod_snr_data.items():
            # X_min = data.min(axis=0)
            # X_max = data.max(axis=0)
            # X_range = X_max - X_min
            # X_range[X_range == 0] = 1
            # normalized_data = 2 * (data - X_min) / X_range - 1

            # Q1 = np.percentile(data, 25, axis=0)
            # Q3 = np.percentile(data, 75, axis=0)
            # IQR = Q3 - Q1
            # IQR[IQR == 0] = 1
            # normalized_data = (data - Q1) / IQR

            # mod, snr = mod_snr
            # if (mod == "WBFM" and snr <= -16):
            #     X_min, X_max = 1, 2
            # elif (mod == "8PSK" and snr <= -16):
            #     X_min, X_max = 2, 3
            # else:
            #     X_min, X_max = 0, 1
            # X_range = X_max - X_min
            # data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
            # normalized_data = X_min + data * X_range
            #
            # X[mod_snr] = normalized_data
        return X

    # Split data ensuring 80% for training and 20% for validation for each mod-snr
    def split_data(self, mod_snr_data):
        mod_snr_train = {}
        mod_snr_val = {}
        train_indices = []
        val_indices = []

        start_idx = 0
        for mod_snr, data in mod_snr_data.items():
            np.random.seed(2024)
            num_examples = data.shape[0]
            num_train = int(num_examples * 0.8)
            indices = np.arange(num_examples)

            # Shuffle indices and split into training and validation
            np.random.shuffle(indices)
            train_idx = indices[:num_train]
            val_idx = indices[num_train:]

            # Print to verify how the data is split
            # print(
            #     f"Mod: {mod_snr[0]}, SNR: {mod_snr[1]}, Total: {num_examples}, Train: {num_train}, Val: {num_examples - num_train}")

            global_train_idx = train_idx + start_idx
            global_val_idx = val_idx + start_idx

            mod_snr_train[mod_snr] = data[train_idx]
            mod_snr_val[mod_snr] = data[val_idx]

            train_indices.extend(global_train_idx)
            val_indices.extend(global_val_idx)

            start_idx += num_examples

        return mod_snr_train, mod_snr_val, train_indices, val_indices

    # One-hot encoding function
    def to_onehot(self, y):
        y_ = np.zeros([len(y), len(self.mods)])
        y_[np.arange(len(y)), [self.mods.index(mod) for mod, _ in y]] = 1
        return y_

    def linear_interpolation(self, data, original_size, new_size):
        x = np.linspace(0, 1, original_size)
        x_new = np.linspace(0, 1, new_size)
        interpolation = np.zeros((data.shape[0], data.shape[1], new_size))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                interpolation[i, j] = np.interp(x_new, x, data[i, j])
        return interpolation

    def load_data(self):
        try:
            with open(self.dataset_path, 'rb') as file:
                self.Xd = pickle.load(file, encoding='iso-8859-1')
        except FileNotFoundError:
            print("The path is incorrect")
            return None
        except Exception as e:
            print(f"The dataset is incorrect: {e}")
            return None

        self.mods, self.snrs = [sorted(list(set([i[j] for i in self.Xd.keys()]))) for j in [0, 1]]

        self.X = []
        self.lbl = []
        self.mod_snr_data = {}
        self.mod_count = {mod: 0 for mod in self.mods}
        self.snr_count = {snr: 0 for snr in self.snrs}

        # Load data and count examples for each mod-snr pair
        for mod in self.mods:
            for snr in self.snrs:
                if (mod, snr) in self.Xd:
                    self.mod_snr_data[(mod, snr)] = self.Xd[(mod, snr)]
                    self.mod_count[mod] += self.Xd[(mod, snr)].shape[0]
                    self.snr_count[snr] += self.Xd[(mod, snr)].shape[0]
                else:
                    print(f"The dataset is missing {(mod, snr)}")

        if not self.mod_snr_data:
            print("No data is loaded")
            return None

        # Flatten the data and labels
        for mod_snr, data in self.mod_snr_data.items():
            self.X.extend(data)
            self.lbl.extend([mod_snr] * data.shape[0])

        self.X = np.vstack(self.X)

        # Preprocess data
        # self.X = self.preprocess_data(self.X, self.mod_snr_data)

        # Perform the split
        mod_snr_train, mod_snr_val, train_indices, val_indices = self.split_data(self.mod_snr_data)

        # Concatenate data into final training and validation sets
        X_train = np.vstack(list(mod_snr_train.values()))
        X_val = np.vstack(list(mod_snr_val.values()))

        # Interpolate data
        # X_train = self.linear_interpolation(X_train, 128, 256)
        # X_val = self.linear_interpolation(X_val, 128, 256)

        lbl_train = [self.lbl[i] for i in train_indices]
        lbl_val = [self.lbl[i] for i in val_indices]

        # One-hot encode labels
        Y_train = self.to_onehot(lbl_train)
        Y_val = self.to_onehot(lbl_val)

        return ((self.mods, self.snrs, self.lbl),
                (X_train, Y_train),
                (X_val, Y_val),
                (train_indices, val_indices))


if __name__ == '__main__':
    dataset_path = "./dataset/RML2016.10a_dict.pkl"
    # dataset_path = "./dataset/RML22Dataset/RML22"
    loader = DatasetLoader(dataset_path)
    data = loader.load_data()

    if data is not None:
        ((mods, snrs, lbl),
         (X_train, Y_train),
         (X_val, Y_val),
         (train_idx, val_idx)) = data
    if data is not None:
        mods, snrs, lbl = data[0]
        X_train, Y_train = data[1]
        _, _ = data[2]
        _, _ = data[3]

        for mod in mods:
            fig, axs = plt.subplots(len(snrs), figsize=(10, 5 * len(snrs)))
            if len(snrs) == 1:
                axs = [axs]

            for i, snr in enumerate(snrs):
                mod_snr_key = (mod, snr)
                if mod_snr_key in loader.mod_snr_data:
                    data_for_plot = loader.mod_snr_data[mod_snr_key][0]

                    axs[i].plot(data_for_plot[0], label='Real', linewidth=0.7)
                    axs[i].plot(data_for_plot[1], '--', label='Imaginary', linewidth=0.7)
                    axs[i].set_title(f'{mod} at {snr} dB')
                    axs[i].legend()

            plt.tight_layout()
            plt.savefig(f'dataset/{mod}.svg')
    for mod in mods:
        if mod == '8PSK':
            for snr in snrs:
                if snr == -20:
                    mod_snr_key = (mod, snr)

                    if mod_snr_key in loader.mod_snr_data:
                        data_for_plot = loader.mod_snr_data[mod_snr_key][0]

                        print(f"First sample data for {mod} at {snr} dB:")
                        print(f"Real part: {data_for_plot[0]}")
                        print(f"Imaginary part: {data_for_plot[1]}")
