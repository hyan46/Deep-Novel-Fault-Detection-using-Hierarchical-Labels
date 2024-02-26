"""The flow of the experiment to test the hypothesis whether
models trained with hierarhcically regularized soft labels
generate more seperated Maximum Softmax Probabilities or not
in comparison to out-of-distributin samples

Case Study: Hot Steel Rolling Dataset
"""

import time
import random
import itertools
from pathlib import Path
from metaflow import FlowSpec, Parameter, step, JSONType, current, namespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18
from sklearn.metrics import confusion_matrix
import torch.optim as optim
from src.utils import (
    make_deterministic,
    TorchCheckpointer,
    get_freer_gpu,
    load_json,
    CSVLogger,
    plot_confusion_matrix,
    tensor2np,
)
from src.tree import (
    HardLabelTransform,
    make_tree_from,
    remove_from_tree,
    SoftlabelTransform,
)
from src.datasets.rolling import (
    generate_lookup_dataframes_for_all_splits,
    RollingDataset,
    train_transforms,
    test_transforms,
)
from src.models.alexnet_rolling import AlexNet
from src.losses import softlabel_crossentropy
from settings import CHECKPOINT_DIR

namespace(None)

def get_criterion(model_type: str):
    criterion = (
        nn.CrossEntropyLoss()
        if model_type == "flat"
        else softlabel_crossentropy
    )
    return criterion


class HypothesisOneFlow_testtime(FlowSpec):
    tree_desc_file_path = Parameter(
        "treeDescFilePath",
        help="where to read hierarchy descripton from",
        default="rolling_hierarchy_description.json",
    )
    left_out = Parameter(
        "leftOut", help="Which class to leave out", required=True
    )
    model_type = Parameter(
        "modelType",
        help="'flat' for flat classifier or 'hier' for hierarhcical",
        required=True,
    )
    data_split_seed = Parameter(
        "dataSplitSeed",
        help="What seed to use to split data into train test val",
        default=42,
    )
    data_split_ratios = Parameter(
        "dataSplitRatios",
        help="Ratios of test-train-split, should sum up to 100 and each of them has to be >0",
        type=JSONType,
        default="[0.6, 0.2, 0.2]",  #original 0.6 0.2 0.2
    )
    beta = Parameter(
        "beta",
        help="Beta of soft-labels, larger -> one-hot, smaller -> uniform",
        default=100.0,
    )
    train_seed = Parameter(
        "trainSeed",
        help="Seed used for model param init + loader shuffle and other things",
        default=42,
    )
    weight_decay = Parameter(
        "weightDecay", help="Weight Decay for Adam Optimizer", default=0.0
    )
    learning_rate = Parameter(
        "learningRate",
        help="Learning rate for Adam optimizer",
        default=3e-4,
    )
    num_epochs = Parameter(
        "numEpochs", help="Number of epochs to train", default=300
    )
    batch_size = Parameter("batchSize", help="Training batch size", default=64)
    pre_sleep = Parameter(
        "preSleep",
        help="Add random sleep between 1 to 20 seconds",
        default=False,
        type=bool,
    )

    @step
    def start(self):
        if self.pre_sleep:
            print("Adding pre sleep")
            time.sleep(random.random() * 20 + 1)
        self.next(self.prepare_dataset)

    @step
    def prepare_dataset(self):
        start_time = time.time()
        self.tree_root = make_tree_from(
            tree_descriptor=load_json(self.tree_desc_file_path)
        )
        self.tree_root = remove_from_tree(self.tree_root, self.left_out)
        self.hard_label_transform = HardLabelTransform(self.tree_root)
        self.soft_label_transform = SoftlabelTransform(
            self.tree_root, self.beta
        )
        self.lookups = generate_lookup_dataframes_for_all_splits(
            left_out=self.left_out,
            split_ratios=self.data_split_ratios,
            split_seed=self.data_split_seed,
        )
        end_time = time.time()  # End time measurement
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Dataset preparation step took {elapsed_time} seconds.")
        self.next(self.train)

    @step
    def train(self):
        start_time = time.time()
        make_deterministic(self.train_seed)
        writer = SummaryWriter(f"runs_testtime/{current.run_id}")
#         device = get_freer_gpu()
        device = 'cuda:1'
        # model = AlexNet(num_classes=len(self.tree_root.leaves))
        model = resnet18(
            num_classes=len(self.tree_root.leaves),
            pretrained=False,   
        )
        model.to(device)
        datasets, loaders = dict(), dict()
        conversion = (
            self.hard_label_transform
            if self.model_type == "flat"
            else self.soft_label_transform
        )
        for partition, transforms, shuffle in zip(
            ["train", "val"],
            [train_transforms(), test_transforms()],
            [True, False],
        ):
            datasets[partition] = RollingDataset(
                self.lookups[partition],
                transforms=transforms,
                label_transforms=conversion,
            )
            loaders[partition] = DataLoader(
                dataset=datasets[partition],
                batch_size=self.batch_size,
                shuffle=shuffle,
            )

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.checkpoint_path = (
            Path(CHECKPOINT_DIR)
            / f"{current.run_id}+{current.task_id}.pth.tar"
        )
        checkpointer = TorchCheckpointer(self.checkpoint_path, True)
        criterion = get_criterion(self.model_type)

        for epoch, is_train in itertools.product(
            range(self.num_epochs), [True, False]
        ):
            if is_train:
                model.train()
                loader = loaders["train"]
                dataset_len = len(datasets["train"])
            else:
                model.eval()
                loader = loaders["val"]
                dataset_len = len(datasets["val"])

            torch.set_grad_enabled(is_train)

            running_loss = 0.0
            running_corrects = 0
            running_labels = []
            running_preds = []
            for batch in loader:
                x, y = batch["image"].to(device), batch["label"].to(device)

                if is_train:
                    optimizer.zero_grad()

                logits = model(x)
                _, preds = torch.max(logits, 1)
                if self.model_type == "hier":
                    _, y_true = torch.max(y, 1)
                    y_labels = y_true.data
                else:
                    y_labels = y.data
                loss = criterion(logits, y)

                if is_train:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == y_labels)
                running_preds.extend(preds.tolist())
                running_labels.extend(y_labels.tolist())

            epoch_loss = running_loss / dataset_len

            # Attempt checkpoint only during validation step, also tensorboard writing
            if not is_train:
                checkpointer.update_if_better(
                    epoch_loss=epoch_loss, epoch=epoch, model=model
                )
                writer.add_scalar("loss", epoch_loss, global_step=epoch)
                writer.add_scalar(
                    "acc", running_corrects / dataset_len, global_step=epoch
                )
                if epoch % 10 == 0:
                    fig = plot_confusion_matrix(
                        confusion_matrix(running_labels, running_preds),
                        self.hard_label_transform.le.classes_,
                    )
                    writer.add_figure("CM", fig, global_step=epoch)
        end_time = time.time()  # End time measurement
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Training step took {elapsed_time} seconds.")
        self.next(self.test)

    @step
    def test(self):
        start_time = time.time()
        # Load checkpointed model from related task
        criterion = get_criterion(self.model_type)
        conversion = (
            self.hard_label_transform
            if self.model_type == "flat"
            else self.soft_label_transform
        )
        loaders, device, model, datasets = self.prepare_for_test(
            label_transforms=conversion
        )
        # Log test classificaiton metrics to a dict
        self.classification_metrics = dict()

        running_loss = 0.0
        running_corrects = 0
        for batch in loaders["test"]:
            x, y = batch["image"].to(device), batch["label"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            _, preds = torch.max(logits, 1)
            if self.model_type == "hier":
                _, y_true = torch.max(y, 1)
                y_labels = y_true.data
            else:
                y_labels = y.data

            running_loss += loss.item()
            running_corrects += torch.sum(preds == y_labels)
        self.classification_metrics["test_loss"] = running_loss / len(
            datasets["test"]
        )
        self.classification_metrics[
            "test_acc"
        ] = running_corrects.item() / len(datasets["test"])
        end_time = time.time()  
        elapsed_time = end_time - start_time  
        print(f"Testing step took {elapsed_time} seconds.")
        self.next(self.log_ood_metrics)

    @step
    def log_ood_metrics(self):
        csv_logger = CSVLogger()
        loaders, device, model, _ = self.prepare_for_test()
        for partition, is_ood in zip(["test", "ood"], [False, True]):
            for batch in loaders[partition]:
                x, y = batch["image"].to(device), batch["label"]
                logits = model(x)
                softmax = torch.log(torch.softmax(logits,1)) #convert Equation 11 to log
                max_softmax, preds = torch.max(softmax, 1)
                for y_i, sm_i, msm_i, pred_i in zip(
                    y, softmax, max_softmax, preds
                ):
                    csv_logger.add_key_value_pair("IsOOD", is_ood)
                    csv_logger.add_key_value_pair("RealID", y_i)
                    pred_i_label = (
                        self.hard_label_transform.le.inverse_transform(
                            [tensor2np(pred_i)]
                        )[0]
                    )
                    csv_logger.add_key_value_pair("PredID", pred_i_label)
                    if self.model_type == "flat":
                        msp = tensor2np(msm_i)
                    else:
                        msp = sum(
                            self.soft_label_transform(pred_i_label)
                            * tensor2np(sm_i)
                        )
                    csv_logger.add_key_value_pair("MSP", msp.item())
        self.ood_logs = csv_logger.to_dataframe()
        self.next(self.end)

    @step
    def end(self):
        pass  # TODO:

    def prepare_for_test(self, label_transforms=None):
        datasets, loaders = dict(), dict()

        for partition in ["test", "ood"]:
            datasets[partition] = RollingDataset(
                self.lookups[partition],
                transforms=test_transforms(),
                label_transforms=label_transforms,
            )
            loaders[partition] = DataLoader(
                dataset=datasets[partition],
                batch_size=self.batch_size,
                shuffle=False,
            )

#         device = get_freer_gpu()
        device = 'cuda:1'

        model = resnet18(
            num_classes=len(self.tree_root.leaves),
            pretrained=False,
        )
        model.load_state_dict(
            torch.load(self.checkpoint_path)["model_state_dict"]
        )
        model.to(device)
        model.eval()
        torch.set_grad_enabled(False)
        return loaders, device, model, datasets


if __name__ == "__main__":
    HypothesisOneFlow_testtime()