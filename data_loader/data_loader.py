import torch
import os
import random
import ast
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from functools import lru_cache
from sklearn.preprocessing import StandardScaler
import pickle


class CachedWindowDataset(Dataset):
    """Precompute window slices once so training does not repeat slicing/copying."""

    def __init__(self, base_dataset):
        self.source_name = type(base_dataset).__name__
        self.mode = getattr(base_dataset, "mode", None)
        self.step = getattr(base_dataset, "step", 1)
        self.win_size = getattr(base_dataset, "win_size", None)
        if hasattr(base_dataset, "train_labels"):
            self.train_labels = getattr(base_dataset, "train_labels")
        if hasattr(base_dataset, "test_labels"):
            self.test_labels = getattr(base_dataset, "test_labels")
        windows = []
        labels = []
        has_labels = None
        for idx in range(len(base_dataset)):
            sample = base_dataset[idx]
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                window, label = sample[0], sample[1]
                if has_labels is None:
                    has_labels = True
                windows.append(np.asarray(window, dtype=np.float32))
                labels.append(np.asarray(label, dtype=np.float32))
            else:
                if has_labels is None:
                    has_labels = False
                windows.append(np.asarray(sample, dtype=np.float32))
        self.windows = np.stack(windows, axis=0).astype(np.float32, copy=False)
        self.labels = (
            np.stack(labels, axis=0).astype(np.float32, copy=False)
            if has_labels
            else None
        )

    def __len__(self):
        return int(self.windows.shape[0])

    def __getitem__(self, index):
        if self.labels is None:
            return self.windows[int(index)]
        return self.windows[int(index)], self.labels[int(index)]


def _parse_anomaly_class_list(raw_value):
    value = str(raw_value).strip()
    if not value or value.lower() == "nan":
        return []
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value.strip():
        return []
    return [token.strip().strip("'\"") for token in value.split(",") if token.strip()]


@lru_cache(maxsize=8)
def _load_nasa_channel_metadata(metadata_path, spacecraft):
    frame = pd.read_csv(metadata_path)
    if spacecraft:
        frame = frame[frame["spacecraft"].astype(str).str.upper() == str(spacecraft).upper()]

    channel_metadata = {}
    for row in frame.to_dict("records"):
        channel_id = str(row["chan_id"])
        anomaly_sequences = ast.literal_eval(str(row["anomaly_sequences"]))
        anomaly_classes = _parse_anomaly_class_list(row["class"])
        num_values = int(row["num_values"])

        item = channel_metadata.setdefault(
            channel_id,
            {
                "spacecraft": str(row["spacecraft"]),
                "num_values": num_values,
                "anomaly_sequences": [],
                "anomaly_classes": [],
            },
        )
        if item["num_values"] != num_values:
            raise ValueError(f"Inconsistent num_values for channel {channel_id}: {item['num_values']} vs {num_values}")

        item["anomaly_sequences"].extend(
            [[int(start), int(end)] for start, end in anomaly_sequences]
        )
        item["anomaly_classes"].extend(anomaly_classes)

    return channel_metadata


def _build_nasa_test_labels(num_values, anomaly_sequences):
    labels = np.zeros(int(num_values), dtype=np.float32)
    if labels.size == 0:
        return labels

    for start, end in anomaly_sequences:
        start = max(0, int(start))
        end = min(labels.size - 1, int(end))
        if end < start:
            continue
        labels[start:end + 1] = 1.0
    return labels


class NASASubdomainSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train", spacecraft="SMAP", entity_id="", metadata_path=""):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.spacecraft = spacecraft
        self.entity_id = str(entity_id).strip()
        self.metadata_path = metadata_path or os.path.join(data_path, "labeled_anomalies.csv")

        if not self.entity_id:
            raise ValueError("entity_id is required for NASASubdomainSegLoader.")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        channel_metadata = _load_nasa_channel_metadata(self.metadata_path, self.spacecraft)
        if self.entity_id not in channel_metadata:
            raise KeyError(f"Channel {self.entity_id} not found in metadata: {self.metadata_path}")

        train_path = os.path.join(data_path, "train", f"{self.entity_id}.npy")
        test_path = os.path.join(data_path, "test", f"{self.entity_id}.npy")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training file not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Testing file not found: {test_path}")

        self.channel_metadata = channel_metadata[self.entity_id]
        self.scaler = StandardScaler()

        train_data = np.load(train_path)
        train_data = np.nan_to_num(train_data)
        self.scaler.fit(train_data)
        self.train = self.scaler.transform(train_data)

        test_data = np.load(test_path)
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.val = self.test
        self.test_labels = _build_nasa_test_labels(
            self.channel_metadata["num_values"],
            self.channel_metadata["anomaly_sequences"],
        )

        if self.test.shape[0] != self.test_labels.shape[0]:
            raise ValueError(
                f"Label length mismatch for {self.entity_id}: "
                f"test={self.test.shape[0]}, labels={self.test_labels.shape[0]}"
            )

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == "val":
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        elif self.mode == "test":
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        else:
            start = index // self.step * self.win_size
            end = start + self.win_size
            return np.float32(self.test[start:end]), np.float32(self.test_labels[start:end])


class GECCOSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/GECCO_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/GECCO_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/GECCO_test_label.npy")
        train_label_path = data_path + "/GECCO_train_label.npy"
        if os.path.exists(train_label_path):
            self.train_labels = np.load(train_label_path)
        else:
            self.train_labels = np.zeros(self.train.shape[0], dtype=np.float32)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(
                self.train_labels[index:index + self.win_size]
            )
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class GenesisSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", scaler_fit_mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/Genesis_train.npy")
        data = np.nan_to_num(data)
        test_data = np.load(data_path + "/Genesis_test.npy")
        test_data = np.nan_to_num(test_data)
        scaler_fit_mode = str(scaler_fit_mode).strip().lower()
        fit_data = test_data if scaler_fit_mode == "test" else data
        self.scaler.fit(fit_data)
        data = self.scaler.transform(data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/Genesis_test_label.npy")
        train_label_path = data_path + "/Genesis_train_label.npy"
        if os.path.exists(train_label_path):
            self.train_labels = np.load(train_label_path)
        else:
            self.train_labels = np.zeros(self.train.shape[0], dtype=np.float32)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(
                self.train_labels[index:index + self.win_size]
            )
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class PUMPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/PUMP_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/PUMP_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/PUMP_test_label.npy")
        train_label_path = data_path + "/PUMP_train_label.npy"
        if os.path.exists(train_label_path):
            self.train_labels = np.load(train_label_path)
        else:
            self.train_labels = np.zeros(self.train.shape[0], dtype=np.float32)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(
                self.train_labels[index:index + self.win_size]
            )
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class PSMSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/PSM_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/PSM_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/PSM_test_label.npy")
        self.train_labels = np.zeros(self.train.shape[0], dtype=np.float32)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(
                self.train_labels[index:index + self.win_size]
            )
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class WaDiSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/WaDi_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/WaDi_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/WaDi_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SKABSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SKAB_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SKAB_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SKAB_test_label.npy")
        train_label_path = data_path + "/SKAB_train_label.npy"
        if os.path.exists(train_label_path):
            self.train_labels = np.load(train_label_path)
        else:
            self.train_labels = np.zeros(self.train.shape[0], dtype=np.float32)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(
                self.train_labels[index:index + self.win_size]
            )
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class HAISegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/HAI_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/HAI_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/HAI_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(
    index,
    data_path,
    batch_size,
    win_size=100,
    step=1,
    mode='train',
    dataset='SKAB',
    entity_id='',
    spacecraft='',
    metadata_path='',
    scaler_fit_mode='train',
    cache_windows=False,
    pin_memory=False,
):
    effective_step = max(1, int(step))
    # Most legacy presets were reported with dense windows. SMAP can opt into
    # a sparse training stride through config.train_step/test_step.
    if (dataset == 'GECCO'):
        dataset = GECCOSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'Genesis'):
        dataset = GenesisSegLoader(data_path, win_size, 1, mode, scaler_fit_mode=scaler_fit_mode)
    elif (dataset == 'PUMP'):
        dataset = PUMPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWAT'):
        dataset = SWATSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'WaDi'):
        dataset = WaDiSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SKAB'):
        dataset = SKABSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'HAI'):
        dataset = HAISegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, effective_step, mode)
    elif (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP_SUBDOMAIN'):
        dataset = NASASubdomainSegLoader(
            data_path,
            win_size,
            1,
            mode,
            spacecraft=spacecraft or 'SMAP',
            entity_id=entity_id,
            metadata_path=metadata_path,
        )
    shuffle = False
    if mode == 'train':
        shuffle = True

    if bool(cache_windows):
        dataset = CachedWindowDataset(dataset)
        size_mb = dataset.windows.nbytes / (1024.0 * 1024.0)
        if dataset.labels is not None:
            size_mb += dataset.labels.nbytes / (1024.0 * 1024.0)
        print(
            f"[DataLoader] Cached {mode} windows | "
            f"source={dataset.source_name} | "
            f"shape={tuple(dataset.windows.shape)} | "
            f"memory={size_mb:.1f} MB"
        )

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0,
                             drop_last=False,
                             pin_memory=bool(pin_memory))
    return data_loader
