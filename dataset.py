# ------------------------------------------------------------------------------------------------
# Transformer
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

        # Perform the split
        mod_snr_train, mod_snr_val, train_indices, val_indices = self.split_data(self.mod_snr_data)

        # Concatenate data into final training and validation sets
        X_train = np.vstack(list(mod_snr_train.values()))
        X_val = np.vstack(list(mod_snr_val.values()))

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
    # dataset_path = "./dataset/RML22/RML22"
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

