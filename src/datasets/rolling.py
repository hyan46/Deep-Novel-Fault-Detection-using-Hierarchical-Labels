from pathlib import Path
from collections import defaultdict
import pandas as pd
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from PIL import Image
from settings import ROLLING_DATA_PATH
from .square_padding import SquarePad
import pdb

def train_transforms():
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def test_transforms():
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def get_all_rolling_df():
    df_logger = defaultdict(list)

    for file_type in ["png", "jpg"]:
        for img_path in Path(ROLLING_DATA_PATH).glob(f"**/*.{file_type}"):
            df_logger["ClassId"].append(img_path.parts[-2])
            df_logger["Path"].append(str(img_path))

    return pd.DataFrame(df_logger).sort_values(by="Path")


class RollingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transforms=None,
        label_transforms=None,
    ):
        self.images = []
        self.labels = []
        for row in df.iterrows():
            self.images.append(Image.open(row[1]["Path"]))
            self.labels.append(row[1]["ClassId"])
        self.transform = transforms
        self.label_transform = label_transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.label_transform:
            label = self.label_transform(label)

        return {"image": image, "label": label}


def generate_lookup_dataframes_for_all_splits(
    split_ratios, split_seed, left_out
):
    all_rolling_df = get_all_rolling_df()
    in_dist_df, trn_df, val_df, tst_df, out_dist_df = split(
        all_rolling_df, split_ratios, split_seed, left_out
    )

    trn_df.reset_index(inplace=True)
    val_df.reset_index(inplace=True)
    tst_df.reset_index(inplace=True)

    return {"train": trn_df, "val": val_df, "test": tst_df, "ood": out_dist_df}


def split(all_rolling_df, split_ratios, split_seed, left_out):
    in_dist_df = all_rolling_df.query("not(ClassId == @left_out)")
    out_dist_df = all_rolling_df.drop(in_dist_df.index)
    trn_df = in_dist_df.groupby("ClassId", group_keys=False).apply(
        lambda x: x.sample(
            int(len(x) * split_ratios[0]), random_state=split_seed
        )
    )
    tmp_df = in_dist_df.drop(trn_df.index)
    val_df = tmp_df.groupby("ClassId", group_keys=False).apply(
        lambda x: x.sample(
            int(
                len(x)
                * (split_ratios[1] / (split_ratios[1] + split_ratios[2]))
            ),
            random_state=split_seed,
        )
    )
    tst_df = tmp_df.drop(val_df.index)
    del tmp_df
    return in_dist_df, trn_df, val_df, tst_df, out_dist_df
