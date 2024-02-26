from collections import defaultdict
from typing import Union
import random
import json
from pathlib import Path
import itertools
import warnings
import math
import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from numpy.random import seed as numpy_seed
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


class CSVLogger:
    def __init__(self) -> None:
        self.data_dict = defaultdict(list)

    def add_key_value_pair(self, key, value) -> None:
        self.data_dict[key].append(value)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data_dict)

    def to_csv(self, out_filepath: Union[str, Path]):
        self.to_dataframe().to_csv(out_filepath, index=False)


class TorchCheckpointer:
    def __init__(
        self,
        output_file: Union[str, Path],
        lower_is_better: bool,
    ) -> None:
        self.lower_is_better = lower_is_better
        open(output_file, "a").close()
        self.output_file = output_file
        self.last_best_value = math.inf if self.lower_is_better else -math.inf
        self.is_better = (
            lambda x: x < self.last_best_value
            if self.lower_is_better
            else lambda x: x > self.last_best_value
        )

    def update_if_better(
        self, epoch_loss: float, epoch: int, model: nn.Module
    ):
        if self.is_better(epoch_loss):
            os.remove(self.output_file)
            self.last_best_value = epoch_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "epoch_loss": epoch_loss,
                },
                self.output_file,
            )


def tensor2np(t):
    return t.cpu().detach().numpy()


def make_deterministic(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    numpy_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    warnings.warn(
        "You have chosen to seed training. "
        "This will turn on the CUDNN deterministic setting, "
        "which can slow down your training considerably! "
        "You may see unexpected behavior when restarting "
        "from checkpoints."
    )


def get_freer_gpu():
    # os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    os.system("nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp")
    memory_available = [
        int(x.split()[2]) for x in open("tmp", "r").readlines()
    ]
    return np.argmax(memory_available)


def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(
        cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2
    )

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def set_paper_style():
    # This sets reasonable defaults for font size for a figure that will go in a paper
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font="serif")

    # Make the background white and specific font family
    sns.set_style(
        "white",
        {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]},
    )

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
