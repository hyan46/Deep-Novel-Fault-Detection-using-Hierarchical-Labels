from collections import defaultdict
import pickle
from pathlib import Path
from copy import deepcopy
from metaflow import Flow, Step
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, Grayscale
import torch
import scipy as sp

from hypothesis1_log import get_criterion
from src.tree import SoftlabelTransform
from src.datasets.rolling import (
    RollingDataset,
    test_transforms,
)
from src.utils import CSVLogger, tensor2np, get_freer_gpu
from src.gda import GaussianDiscriminantAnalysis
from settings import DERIVATIVES_DIR

to_pil = ToPILImage()
to_gray = Grayscale()


def prepare_dataset(tree_root, lookups, checkpoint_path, device):
    datasets, loaders = dict(), dict()

    for partition in ["test", "ood", "val", "train"]:
        datasets[partition] = RollingDataset(
            lookups[partition],
            transforms=test_transforms(),
            label_transforms=None,
        )
        loaders[partition] = DataLoader(
            dataset=datasets[partition],
            batch_size=32,
            shuffle=False,
        )

    model = resnet18(
        num_classes=len(tree_root.leaves),
        pretrained=False,
    )
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # torch.set_grad_enabled(False)
    return loaders, model, datasets


def steps_df():
    flow = Flow("HypothesisOneFlow_testtime")
    runs = list(flow)
    entries = []
    for run in runs:
        try:
            step = Step(f"HypothesisOneFlow_log_0221/{run.id}/log_ood_metrics")
            entry = deepcopy(step.task.data.classification_metrics.copy())
            entry["run_id"] = run.id
            entry["left_out"] = step.task.data.left_out
            entry["learning_rate"] = step.task.data.learning_rate
            entry["weight_decay"] = step.task.data.weight_decay
            entry["model_type"] = step.task.data.model_type
            entry["train_seed"] = step.task.data.train_seed
            entry["num_epochs"] = step.task.data.num_epochs
            entry["beta"] = step.task.data.beta
            df = step.task.data.ood_logs
            entry["auroc"] = roc_auc_score(df["IsOOD"], -df["MSP"])
            entries.append(entry)
        except:
            print(f"skipped {run.id}")
        else:
            pass
    return pd.DataFrame(entries)


def step_from_run_id(run_id):
    return Step(f"HypothesisOneFlow_testtime/{run_id}/log_ood_metrics")


def step_essentials(step, device):
    loaders, model, datasets = prepare_dataset(
        step.task.data.tree_root,
        step.task.data.lookups,
        step.task.data.checkpoint_path,
        device,
    )

    hard_label_transform = step.task.data.hard_label_transform
    soft_label_transform = step.task.data.soft_label_transform
    return loaders, model, datasets, hard_label_transform, soft_label_transform


def iterate_logits_labels_for_run_id(run_id, partition):
    step = Step(f"HypothesisOneFlow_log_0221/{run_id}/log_ood_metrics")
    device = 'cuda:1'
    (
        loaders,
        model,
        datasets,
        hard_label_transform,
        soft_label_transform,
    ) = step_essentials(step, device)

    for batch in loaders[partition]:
        image, batch_labels = batch["image"].to(device), batch["label"]
        batch_logits = model(image)
        _, preds = torch.max(batch_logits, 1)
        for logit, real_label, pred in zip(batch_logits, batch_labels, preds):
            pred_label = hard_label_transform.le.inverse_transform(
                [pred.item()]
            )[0]
            soft_label = soft_label_transform(pred_label)
            yield tensor2np(logit), real_label, pred_label, soft_label


            
def pickle_test_and_ood_logits():
    df = steps_df()
    to_pickle = dict()
    for run_id in df.run_id:
        to_pickle[run_id] = dict()
        to_pickle[run_id]["test"] = list()
        to_pickle[run_id]["ood"] = list()
        for partition in ["test", "ood"]:
            for (
                logits,
                real_label,
                pred_label,
                soft_label,
            ) in iterate_logits_labels_for_run_id(
                run_id=run_id, partition=partition
            ):
                entry = {
                    "logits": logits,
                    "soft": soft_label,
                    "real_label": real_label,
                    "pred_label": pred_label,
                }
                to_pickle[run_id][partition].append(entry)

    with open(Path(DERIVATIVES_DIR) / "logits.pkl", "wb") as f:
        pickle.dump(to_pickle, f)
    return df

def pickle_empirical_class_mean_and_cov_of_test_and_val_splits():
    df = steps_df()
    to_pickle = dict()
    for run_id in df.run_id:
        features = defaultdict(list)
        for partition in ["train", "val"]:
            for (
                logits,
                real_label,
                _,
                _,
            ) in iterate_logits_labels_for_run_id(
                run_id=run_id, partition=partition
            ):
                features[real_label].append(logits)
        features = {
            class_id: np.stack(list_of_logits)
            for class_id, list_of_logits in features.items()
        }

        gda = GaussianDiscriminantAnalysis(features)
        to_pickle[run_id] = gda
    with open(Path(DERIVATIVES_DIR) / "GDAs.pkl", "wb") as f:
        pickle.dump(to_pickle, f)




        

        
def make_dmd_summary():
    with open(Path(DERIVATIVES_DIR) / "GDAs.pkl", "rb") as f:
        gdas = pickle.load(f)
    df = steps_df()
    df["auroc"] = -1.0
    df.set_index("run_id", inplace=True)
    for run_id in df.index:
        gda = gdas[run_id]
        scores = []
        labels = []
        for partition in ["test", "ood"]:
            for (
                logits,
                real_label,
                _,
                _,
            ) in iterate_logits_labels_for_run_id(
                run_id=run_id, partition=partition
            ):
                labels.append(partition == "ood")
                scores.append(gda.mahalanobis_score(logits))
        df.loc[run_id, "auroc"] = roc_auc_score(labels, scores)
    df.to_csv(Path(DERIVATIVES_DIR) / "dmd_summary_final.csv")


def make_odin_summary():
    condition = [ 'A12','A40','A61','A31']
    df = steps_df()
    df.drop_duplicates(subset=['run_id'])
    df.set_index("run_id", inplace=True)
    selected_rows = df['left_out'].isin(condition)
    df = df[selected_rows]
    df.to_csv(Path(DERIVATIVES_DIR) / "msp_summary_final.csv")
    df["auroc"] = -1.0
    device = 'get_freer_gpu()'
    T = 1000
    eps = 0.0012


    for run_id, model_type in zip(df.index, df.model_type):
        run_anomaly_scores = []
        run_labels = []
        criterion = get_criterion(model_type)
        step = Step(f"HypothesisOneFlow_log_0221/{run_id}/log_ood_metrics")
        (
            loaders,
            model,
            _,
            _,
            soft_label_transform,
        ) = step_essentials(step, device)
        for partition in ["test", "ood"]:
            for batch in loaders[partition]:
                image, _ = batch["image"].to(device), batch["label"]
                image.requires_grad = True
                batch_logits = model(image)
                _, preds = torch.max(batch_logits, 1)
                if model_type == "flat":
                    y = preds
                else:
                    y = torch.stack(
                        [
                            torch.from_numpy(
                                soft_label_transform.softmax[idx, :]
                            ).to(device)
                            for idx in tensor2np(preds)
                        ]
                    )

                loss = criterion(batch_logits / T, y)
                loss.backward()

                gradient = torch.ge(image.grad.data, 0)
                gradient = (gradient.float() - 0.5) * 2
                image.requires_grad = False
                batch_logits = model(image - eps * gradient) / T

                if model_type == "flat":
                    scores, _ = torch.max(torch.log(torch.softmax(batch_logits, 1)), 1)
                    scores = tensor2np(-scores).flatten().tolist()
                else:
                    scores = (
                        tensor2np(
                            -torch.sum(torch.log(torch.softmax(batch_logits, 1)) * y, 1)
                        )
                        .flatten()
                        .tolist()
                    )
                run_anomaly_scores.extend(scores)
                run_labels.extend([partition == "ood"] * len(scores))
        df.loc[run_id, "auroc"] = roc_auc_score(run_labels, run_anomaly_scores)
    df.to_csv(Path(DERIVATIVES_DIR) / "odin_summary_final.csv")

def logits_test_df_process():
    condition = [ 'A12','A40','A61','A31']
    df= pd.read_csv(Path(DERIVATIVES_DIR) / "dmd_summary_final.csv")
    print('length1:',len(df))
    df.drop_duplicates(subset=['run_id'])
    selected_rows = df['left_out'].isin(condition)
    df = df[selected_rows]    
    learning_rate_set = [0.0001]
    seed_set = [3]
    df = df[df['learning_rate'].isin(learning_rate_set) ]
    df = df[df['train_seed'].isin(seed_set) ]
    df_flat = df[df['model_type'] == 'flat']
    df_hier = df[df['model_type'] == 'hier']
    beta_set = [10]
    df_hier = df_hier[df_hier['beta'].isin(beta_set) ]
    def process_df(df, model_type):
        df.set_index("run_id", inplace=True)
        logits_dict = defaultdict(dict)
        for run_id in df.index:
            print('*****')
            print(run_id)
            features = defaultdict(list)
            for partition in ["test", "ood"]:
                for (
                    logits,
                    real_label,
                    _,
                    _,
                ) in iterate_logits_labels_for_run_id(
                    run_id=run_id, partition=partition
                ):
                    features[real_label].append(logits)
            features = {
                class_id: np.stack(list_of_logits)
                for class_id, list_of_logits in features.items()
            }
            id = df.at[run_id, 'left_out']
            logits_dict[id] = dict(features)
        with open(Path(DERIVATIVES_DIR) / f"logits_dict_{model_type}.pkl", "wb") as f:
            pickle.dump(logits_dict, f)
    
    # Process each dataframe separately
    if not df_flat.empty:
        process_df(df_flat, 'flat')
    if not df_hier.empty:
        process_df(df_hier, 'hier')



if __name__ == "__main__":
    pickle_test_and_ood_logits()
    pickle_empirical_class_mean_and_cov_of_test_and_val_splits()
    make_dmd_summary()
    make_odin_summary()
    logits_test_df_process()
    
