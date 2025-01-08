# ------------------------------------------------------------------------------------------------
# Transformer
# Copyright (c) 2024 Fanglin Liu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------

import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from models.CNN import ResNet
import dataset
import time
from thop import profile
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from torchinfo import summary
import torch.nn.functional as F
from models.RML_ResNet import RML_ResNet
from models.RML_LSTM import RML_LSTM
from models.RML_DCNet import RML_DCNet
from models.RML_CGDNet import RML_CGDNet
from models.RML_CLDNet import RML_CLDNet
from models.RML_CLDNet2_0 import RML_CLDNet2_0
from models.RML_CNN import RML_CNN
from models.RML_CNN2_0 import RML_CNN2_0
from models.RML_DAE import RML_DAE
from models.RML_DenseNet import RML_DenseNet
from models.RML_GRU2_0 import RML_GRU2_0
from models.RML_IC_AMCNet import RML_IC_AMCNet
from models.RML_MCLDNet import RML_MCLDNet
from models.RML_MCNet import RML_MCNet
from models.RML_PET_CGDNet import RML_PET_CGDNet
import pycwt as wavelet
from matplotlib.ticker import ScalarFormatter
from pycwt.helpers import find
from matplotlib.pyplot import MultipleLocator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
dataset_path = "./dataset/RML2016.10a_dict.pkl"
data_name = "RadioML 2016.10a"
# dataset_path = "./dataset/RML22/RML22"
# data_name = "RML22"
loader = dataset.DatasetLoader(dataset_path)
data = loader.load_data()
((mods, snrs, lbl),
 (X_train, Y_train),
 (X_val, Y_val),
 (train_idx, val_idx)) = data
X_train = np.expand_dims(X_train, axis=1)
X_val = np.expand_dims(X_val, axis=1)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(np.array(X_train))
Y_train = torch.LongTensor(np.array(Y_train)).float()
X_val = torch.FloatTensor(np.array(X_val))
Y_val = torch.LongTensor(np.array(Y_val)).float()

# All hyperparameters
epochs = 96
# epochs = 100
batch_size = 128
learning_rate = 0.001
weight_decay = 0.0001
spatial_shapes = [(2, 128)]
channel = 64
dim_feedforward = 256
input_shape = (1, 2, 128)
classes = 11
dropout = 0.1
activation = "relu"
# activation = "gelu"
num_encoder_layers = 6
num_head = 8
num_feat_levels = 1
enc_n_points = 8

# End

# Create DataLoader
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)


class CustomLRScheduler:
    def __init__(self, optimizer, total_epochs, start_epoch, min_lr, max_lr):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.start_epoch = start_epoch
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_epoch = 0

    def get_lr(self):
        if self.current_epoch <= self.start_epoch:
            lr = self.max_lr
        else:
            lr = self.min_lr
            # progress = (self.current_epoch - self.start_epoch) / (self.total_epochs - self.start_epoch)
            # lr = self.max_lr * (self.min_lr / self.max_lr) ** progress
            # lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        return lr

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_epoch += 1
        print(f"Learning Rate = {lr}")


# Model, loss function, optimizer
# model = RML_ResNet().to(device)
# model = RML_LSTM().to(device)
# model = RML_DCNet().to(device)
# model = RML_CGDNet().to(device)
# model = RML_CLDNet().to(device)
# model = RML_CLDNet2_0().to(device)
# model = RML_CNN().to(device)
# model = RML_CNN2_0().to(device)
# model = RML_DAE().to(device)
# model = RML_DenseNet().to(device)
# model = RML_GRU2_0().to(device)
# model = RML_IC_AMCNet().to(device)
# model = RML_MCLDNet().to(device)
# model = RML_MCNet().to(device)
# model = RML_PET_CGDNet().to(device)
model = ResNet(spatial_shapes=spatial_shapes,
               channel=channel,
               dim_feedforward=dim_feedforward,
               input_shape=input_shape,
               classes=classes,
               dropout=dropout,
               activation=activation,
               num_encoder_layers=num_encoder_layers,
               num_head=num_head,
               num_feat_levels=num_feat_levels,
               enc_n_points=enc_n_points).to(device)
sampling_offsets_params = []
for layer in model.inae.encoder.layers:
    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'sampling_offsets'):
        sampling_offsets_params.extend(layer.self_attn.sampling_offsets.parameters())
optimizer = optim.Adam([
    {'params': sampling_offsets_params, 'lr': learning_rate * 0.1,
     'weight_decay': 0.0001},
    {'params': [p for n, p in model.named_parameters() if 'sampling_offsets' not in n], 'lr': learning_rate,
     'weight_decay': 0.0001}])
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = CustomLRScheduler(optimizer,
                              total_epochs=epochs,
                              start_epoch=40,
                              min_lr=learning_rate * 0.1,
                              max_lr=learning_rate)
criterion = nn.CrossEntropyLoss().to(device)


def model_details():
    global j
    for i in range(10000):
        if not os.path.exists(f'result/exp{i}'):
            os.makedirs(f'result/exp{i}')
            j = i
            break

    print(f"{data_name}")
    print("Hyperparameters:")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")
    print(f"weight_decay: {weight_decay}")
    print(f"spatial_shapes: {spatial_shapes}")
    print(f"channel: {channel}")
    print(f"dim_feedforward: {dim_feedforward}")
    print(f"dropout: {dropout}")
    print(f"activation: {activation}")
    print(f"num_encoder_layers: {num_encoder_layers}")
    print(f"num_head: {num_head}")
    print(f"num_feat_levels: {num_feat_levels}")
    print(f"enc_n_points: {enc_n_points}")
    print(f"optimizer: {optimizer}")

    input = torch.randn(batch_size, X_train.shape[1], X_train.shape[2], X_train.shape[3]).to(device)
    flops, params = profile(model, inputs=(input, ))
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Parameters: {params / 1e6:.2f} M")
    print('a. Model layers:')
    for layer in model.children():
        print(layer)
    print('b. Model layers:')
    summary(model, input_size=(batch_size, X_train.shape[1], X_train.shape[2], X_train.shape[3]))


model_details()


# L1 regularization
def calc_loss(pred, target, l1_lambda):
    loss = criterion(pred, target)
    l1_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l1_reg += (torch.sum(torch.abs(param)))
    loss += (l1_lambda * l1_reg)
    return loss


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.new_zeros(N, C)
        class_mask = class_mask.scatter_(1, targets.unsqueeze(1), 1.)

        if self.alpha >= 0:
            at = self.alpha
        else:
            at = class_mask * -self.alpha

        pt = P * class_mask + (1 - P) * (1 - class_mask)
        pt = pt.clamp(min=0.0001, max=1 - 0.0001)

        loss = -at * (1 - pt) ** self.gamma * torch.log(pt)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# focal_loss = FocalLoss(alpha=0.25, gamma=2, reduction='mean')


# Training loop
def train_model(model, train_loader, val_loader, num_epoch):
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    delta = np.random.normal(loc=0, scale=1e-6)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    batch_val_loss = []
    for epoch in range(num_epoch):
        start_time = time.time()
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for data, target in train_loader:
            data = data.to(device)
            target = target.argmax(dim=1).to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            # loss = calc_loss(output, target, l1_lambda=(1e-5 + delta))
            # loss = focal_loss(output, target).to(device)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        scheduler.step()

        train_loss /= len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Train_loss: {train_loss}, Train_accuracy: {train_accuracy}")

        # with torch.no_grad():
        #     for name, param in model.fc3.named_parameters():
        #         print(f"Layer {name}, param: {param.data}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        sample_times = []
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.argmax(dim=1).to(device)

                start_sample_time = time.time()
                output = model(data)
                end_sample_time = time.time()
                sample_times.append((end_sample_time - start_sample_time) * 1000)

                loss = criterion(output, target)
                # loss = focal_loss(output, target).to(device)
                # batch_val_loss.append(loss)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Val_loss: {val_loss}, Val_accuracy: {val_accuracy}")

        average_sample_time = sum(sample_times) / len(sample_times) if sample_times else 0
        print(f"Average inference time per sample: {average_sample_time:.2f} ms")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'weights/weights.pth')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'weights/weights_acc.pth')

        if epoch == 0:
            torch.save(model.state_dict(), f'weights/weights_{epoch + 1}.pth')

        end_time = time.time()
        run_time = (end_time - start_time) / 60.0
        print(f"Time: {run_time}m")

    # batch_val_loss = torch.tensor(batch_val_loss).cpu().numpy()
    # plt.figure(figsize=(10, 8))
    # plt.plot(batch_val_loss, label='Validation Loss')
    # plt.title('Validation Loss per Batch')
    # plt.xlabel('Batch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs', fontsize=12, fontname='Euclid')
    plt.xlabel('Epoch', fontsize=10, fontname='Euclid')
    plt.ylabel('Accuracy', fontsize=10, fontname='Euclid')
    plt.legend(
        loc='best',
        fontsize=10,
        prop={'family': 'Euclid'},
        frameon=True,
        facecolor='white',
        edgecolor='none',
        framealpha=1.0
    )

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs', fontsize=12, fontname='Euclid')
    plt.xlabel('Epoch', fontsize=10, fontname='Euclid')
    plt.ylabel('Loss', fontsize=10, fontname='Euclid')
    plt.legend(
        loc='best',
        fontsize=10,
        prop={'family': 'Euclid'},
        frameon=True,
        facecolor='white',
        edgecolor='none',
        framealpha=1.0
    )

    plt.tight_layout()
    plt.savefig(f'result/exp{j}/Accuracy_and_Loss.svg')


# train_model(model, train_loader, val_loader, epochs)


def plot_(model, data_loader, val_idx, device):
    plt.rcParams['font.family'] = 'Euclid'
    plt.rcParams['font.size'] = 10
    shutil.copy('weights/weights.pth', f'result/exp{j}/weights.pth')
    shutil.copy('weights/weights_acc.pth', f'result/exp{j}/weights_acc.pth')
    shutil.copy('weights/weights_1.pth', f'result/exp{j}/weights_1.pth')

    model.load_state_dict(torch.load(f'result/exp{j}/weights.pth', map_location=device))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_preds = np.array(all_preds)
    test_labels = np.array(all_labels)
    test_labels = np.argmax(test_labels, axis=1)

    mods = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', '4-PAM', '16-QAM', '64-QAM', 'QPSK', 'WBFM']
    snrs_X_val = [lbl[idx][1] for idx in val_idx]
    unique_snrs = np.unique(snrs_X_val)
    # if 20 in unique_snrs:
    #     unique_snrs = np.delete(unique_snrs, np.where(unique_snrs == 20)[0])

    mod_accuracies = {mod: [] for mod in mods}
    snr_accuracies = {}

    for snr in unique_snrs:
        indices = np.where(np.array(snrs_X_val) == snr)[0]
        if len(indices) > 0:
            current_test_preds = test_preds[indices]
            current_test_labels = test_labels[indices]

            for mod_idx, mod in enumerate(mods):
                mod_indices = np.where(np.array(current_test_labels) == mod_idx)[0]
                if len(mod_indices) > 0:
                    mod_preds = current_test_preds[mod_indices]
                    mod_labels = current_test_labels[mod_indices]
                    accuracy = accuracy_score(mod_labels, mod_preds)
                    mod_accuracies[mod].append(accuracy)
                    if snr not in snr_accuracies:
                        snr_accuracies[snr] = {mod: [] for mod in mods}
                    snr_accuracies[snr][mod] = accuracy

    snr_mean_accuracies = {snr: np.mean([accuracy for mod, accuracy in mod_acc.items()]) for snr, mod_acc in
                           snr_accuracies.items()}
    for snr, mean_accuracy in snr_mean_accuracies.items():
        print(f'SNR={snr} dB: Mean Accuracy = {mean_accuracy * 100:.2f}')

    mod_mean_accuracies = {mod: np.mean(accuracies) for mod, accuracies in mod_accuracies.items()}
    for mod, mean_accuracy in mod_mean_accuracies.items():
        print(f'Modulation {mod}: Mean Accuracy = {mean_accuracy * 100:.2f}')

    plt.figure(figsize=(8, 6))
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'h', 'H', '+']
    for i, (mod, accuracies) in enumerate(mod_accuracies.items()):
        plt.plot(unique_snrs, accuracies, label=mod, marker=markers[i % len(markers)], markersize=4, linewidth=0.7)

    plt.xlabel('SNR (dB)', fontsize=16, fontname='Euclid')  # SimSun
    plt.ylabel('准确率', fontsize=16, fontname='SimSun')
    plt.title(f'{data_name}', fontsize=20, fontname='Euclid')
    for line in plt.gca().lines:
        line.set_zorder(3)
    legend = plt.legend(
        loc='best',
        fontsize=10,
        prop={'family': 'Euclid'},
        frameon=True,
        facecolor='white',
        edgecolor='none',
        framealpha=1.0
    )
    legend.set_zorder(2)
    plt.grid(True, color='gray', linestyle='--', linewidth=0.3)
    if data_name == "RadioML 2016.10a":
        plt.xticks(np.arange(-20, 19, 4), fontsize=16, fontname='Euclid')
    else:
        plt.xticks(np.arange(-20, 21, 4), fontsize=16, fontname='Euclid')
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=16, fontname='Euclid')
    plt.savefig(f'result/exp{j}/Accuracy.svg', format='svg', bbox_inches='tight')
    plt.close()
    print("Accuracy Plot End")

    plt.rcParams['font.family'] = 'Euclid'
    plt.rcParams['font.size'] = 16
    for snr in unique_snrs:
        indices = np.where(np.array(snrs_X_val) == snr)[0]
        if len(indices) > 0:
            current_test_preds = test_preds[indices]
            current_test_labels = test_labels[indices]
            conf_matrix = confusion_matrix(current_test_labels, current_test_preds, labels=range(len(mods)))

            normalized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
            normalized_conf_matrix_rounded = np.around(normalized_conf_matrix, 2)

            plt.figure(figsize=(8, 6))
            # Blues, Reds, Oranges
            annot = np.where(normalized_conf_matrix_rounded >= 0.1, normalized_conf_matrix_rounded, '')
            ax = sns.heatmap(normalized_conf_matrix_rounded,
                        annot=annot, cmap='Reds', fmt="", xticklabels=mods, yticklabels=mods, annot_kws={"fontfamily": "Euclid"})
            plt.xticks(rotation=45, fontsize=16, fontname='Euclid')
            plt.yticks(rotation=45, fontsize=16, fontname='Euclid')
            plt.xlabel('预测标签', fontsize=20, fontname='SimSun')  # fontname='Arial'
            plt.ylabel('真实标签', fontsize=20, fontname='SimSun')
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=20)
            for label in cbar.ax.get_yticklabels():
                label.set_fontname('Euclid')
                label.set_fontsize(16)
            mean_accuracy = snr_mean_accuracies.get(snr, 0)
            plt.title(f'准确率={mean_accuracy * 100:.2f}%, SNR={snr}dB', fontsize=24, fontname='SimSun')
            plt.savefig(f'result/exp{j}/Confusion_Matrix_{snr}.svg', format='svg', bbox_inches='tight')
            # plt.show()
            plt.close()
    print("Confusion Matrix Plot End")

    df_accuracies = pd.DataFrame(index=mods, columns=unique_snrs)

    for mod, accuracies in mod_accuracies.items():
        for snr, accuracy in zip(unique_snrs, accuracies):
            df_accuracies.at[mod, snr] = accuracy

    mod_mean_accuracies = df_accuracies.mean(axis=1)
    df_accuracies['Mean Modulation'] = mod_mean_accuracies

    snr_mean_accuracies = df_accuracies.mean(axis=0)
    df_accuracies.loc['Mean SNR'] = snr_mean_accuracies

    overall_mean_accuracy = df_accuracies.mean().mean()
    df_accuracies.loc['Mean SNR', 'Mean Modulation'] = overall_mean_accuracy
    # df_accuracies.loc['Mean Modulation', 'Mean SNR'] = df_accuracies['Mean SNR'].mean()

    excel_file = 'result/exp{}/accuracies.xlsx'.format(j)
    df_accuracies.to_excel(excel_file)

    df_accuracies_transposed = df_accuracies.T
    transposed_excel_file = f'result/exp{j}/accuracies_transposed.xlsx'
    df_accuracies_transposed.to_excel(transposed_excel_file)
    print(f"Result saved to 'result/exp{j}")


plot_(model, val_loader, val_idx, device)


def create_and_write_file(file_path, content):
    with open(file_path, 'w') as f:
        f.write(content)


file_path = f'result/exp{j}/result.txt'
absolute_file_path = os.path.abspath(file_path)
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
content = f'Current time: {current_time}\n'
create_and_write_file(file_path, content)
if os.name == 'nt':
    try:
        os.startfile(absolute_file_path)
    except Exception as e:
        print(f"Windows Error opening file: {e}")
if os.name == 'posix':
    try:
        os.system(f'xdg-open "{absolute_file_path}"')
    except Exception as e:
        print(f"Linux Error opening file: {e}")
