from copy import deepcopy
from pathlib import Path
import pickle
from src.utils import get_freer_gpu
from metaflow import Flow, Step, namespace
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, accuracy_score
import pandas as pd
import numpy as np
import scipy as sp
import streamlit as st
from torchvision.models import resnet18
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, Grayscale
from anytree.exporter import DotExporter
import pdb
from src.datasets.rolling import (
    RollingDataset,
    test_transforms,
)
from src.utils import tensor2np, CSVLogger, set_paper_style
from src.tree import lca_dist, SoftlabelTransform
from settings import DERIVATIVES_DIR, REPORT_DIR
import pdb
to_pil = ToPILImage()
to_gray = Grayscale()

namespace(None)

beta_set  = [10,100]
condition = [ 'A12','A40','A61','A31']

def prepare_dataset(tree_root, lookups, checkpoint_path, device):
    datasets, loaders = dict(), dict()
    
    for partition in ["test", "ood"]:
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
    torch.set_grad_enabled(False)
    return loaders, model, datasets


def step_graphviz_chart(step):
    st.graphviz_chart(
        "\n".join([line for line in DotExporter(step.task.data.tree_root)])
    )


@st.cache
def make_rank_dist_data(run_id):
    step = Step(f"HypothesisOneFlow_testtime/{run_id}/log_ood_metrics")
#     pdb.set_trace()
    device = get_freer_gpu()
    loaders, model, datasets = prepare_dataset(
        step.task.data.tree_root,
        step.task.data.lookups,
        step.task.data.checkpoint_path,
        device,
    )

    data = pd.DataFrame(
        columns=["LCA_Dist", "Rank", "IsOOD"],
    )

    dict = {False : 'Normal', True: 'Abonormal'}
    data=data.replace({"IsOOD": dict})

    i = 0

    for batch in loaders["test"]:
        image, labels = batch["image"].to(device), batch["label"]
        logits = model(image)
        for row, label in zip(logits, labels):
            argsort = np.argsort(tensor2np(row))
            predicted_name = (
                step.task.data.hard_label_transform.le.inverse_transform(
                    [argsort[-1]]
                )[0]
            )
            rank = 2
            argsort_inverted = argsort[::-1]
            for sort_id in argsort_inverted[1:]:
                rank_name = (
                    step.task.data.hard_label_transform.le.inverse_transform(
                        [sort_id]
                    )[0]
                )
                lca = lca_dist(
                    tree=step.task.data.tree_root,
                    node1_name=predicted_name,
                    node2_name=rank_name,
                )
                data.loc[i] = [lca, rank, False]
                rank += 1
                i += 1

    for batch in loaders["ood"]:
        image, labels = batch["image"].to(device), batch["label"]
        logits = model(image)
        for row, label in zip(logits, labels):
            argsort = np.argsort(tensor2np(row))
            predicted_name = (
                step.task.data.hard_label_transform.le.inverse_transform(
                    [argsort[-1]]
                )[0]
            )
            rank = 2
            argsort_inverted = argsort[::-1]
            for sort_id in argsort_inverted[1:]:
                rank_name = (
                    step.task.data.hard_label_transform.le.inverse_transform(
                        [sort_id]
                    )[0]
                )
                lca = lca_dist(
                    tree=step.task.data.tree_root,
                    node1_name=predicted_name,
                    node2_name=rank_name,
                )
                data.loc[i] = [lca, rank, True]
                rank += 1
                i += 1
    return data


# @st.cache
def steps_df():
    flow = Flow("HypothesisOneFlow_testtime")
    runs_log  = list(flow)
    entries = []
    for run in runs_log :
        try:
            step = Step(f"HypothesisOneFlow_testtime/{run.id}/log_ood_metrics")
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


def make_rank_dist_plots(df):
    left_out = st.sidebar.selectbox("Left Out", options=df.left_out.unique())
    hier_id = st.sidebar.selectbox(
        "Hier Index",
        options=df.query(
            "model_type == 'hier' and left_out == @left_out"
        ).index,
    )
    
    hier_data = make_rank_dist_data(df.iloc[hier_id, :].run_id)
    hier_data_copy = hier_data.copy(deep=True)
    hier_data_copy["model_type"] = "Hier"

    flat_id = st.sidebar.selectbox(
        "Flat Index",
        options=df.query(
            "model_type == 'flat' and left_out == @left_out"
        ).index,
    )

    flat_data = make_rank_dist_data(df.iloc[flat_id, :].run_id)
    flat_data_copy = flat_data.copy()
    flat_data_copy["model_type"] = "Flat"
    data = pd.concat([flat_data_copy, hier_data_copy])
    data.rename(
        columns={
            "LCA_Dist": "Distance to Predicted",
            "model_type": "Model Type",
            "IsOOD": "Is Novel",
            "Rank": "Prediction Rank",
        },
        inplace=True,
    )

    st.header("LCA Dist to Predicted Label by Rank")
    g = sns.relplot(
        y="Distance to Predicted",
        x="Prediction Rank",
        hue="Is Novel",
        # hue_order = ["Normal","Abnormal"],
        kind="line",
        col="Model Type",
        # err_style="bars",
        data=data,
        palette=["green", "purple"],
    )
    g.set(xticks=range(2, 14))
    st.pyplot(g)
    if 10 in beta_set:
        g.savefig(
    #         Path.cwd() / "report_outputs" / f"{df.iloc[flat_id, :].left_out}.pdf"
            Path(REPORT_DIR) / f"{df.iloc[flat_id, :].left_out}.pdf"

        )
    return g


def only_msp_fig(df, logits):
    logger = CSVLogger()
    for run_id, run_dict in logits.items():
        for partition_id, partition_list in run_dict.items():
            is_ood = partition_id == "ood"
            for instance_dict in partition_list:
                if 'withLog' in DERIVATIVES_DIR:
                    max_softmax = np.max(
                        sp.special.log_softmax(instance_dict["logits"])
                    )
                else: 
                    max_softmax = np.max(
                    sp.special.softmax(instance_dict["logits"])
                )
                logger.add_key_value_pair("MaxSoftmax", max_softmax)
                logger.add_key_value_pair("IsOOD", is_ood)
                logger.add_key_value_pair("run_id", str(run_id))
                logger.add_key_value_pair(
                    "real_label", instance_dict["real_label"]
                )
                logger.add_key_value_pair(
                    "pred_label", instance_dict["pred_label"]
                )
    joined = (
        logger.to_dataframe()
        .set_index("run_id")
        .join(df.set_index("run_id")[["left_out", "model_type"]], on="run_id")
    )

    only_msp = (
        joined.groupby(["run_id", "model_type", "left_out"])
        .apply(lambda x: roc_auc_score(x.IsOOD, -x.MaxSoftmax))
        .reset_index(["run_id", "model_type", "left_out"])
    )
    only_msp.rename({0: "auroc"}, inplace=True, axis="columns")
    only_msp = (
        only_msp.groupby(["left_out", "model_type"])["auroc"]
        .nlargest(30)
        .reset_index(["left_out", "model_type"])
    )
    g = sns.catplot(
        x="model_type",
        y="auroc",
        col="left_out",
        col_wrap=4,
        kind="box",
        sharey=False,
        data=only_msp,
    )
    st.write(g.fig)


@st.cache
def get_betas_df(df):
    betas = np.logspace(-3, 3, num=7)
    logger = CSVLogger()
    for beta, beta_id in zip(betas, range(-3, 4)):
        for run_id, run_dict in logits.items():
            slt = SoftlabelTransform(
                Step(
                    f"HypothesisOneFlow_testtime/{run_id}/log_ood_metrics"
                ).task.data.tree_root,
                beta=beta,
            )
            for partition_id, partition_list in run_dict.items():
                is_ood = partition_id == "ood"
                for instance_dict in partition_list:
                    if 'withLog' in DERIVATIVES_DIR:
                        softmax = sp.special.log_softmax(instance_dict["logits"])
                    else:
                        softmax = sp.special.softmax(instance_dict["logits"])
                    score = sum(slt(instance_dict["pred_label"]) * softmax)
                    logger.add_key_value_pair("Beta", beta_id)
                    logger.add_key_value_pair("Score", score)
                    logger.add_key_value_pair("IsOOD", is_ood)
                    logger.add_key_value_pair("run_id", str(run_id))
                    logger.add_key_value_pair(
                        "real_label", instance_dict["real_label"]
                    )
                    logger.add_key_value_pair(
                        "pred_label", instance_dict["pred_label"]
                    )
    st.write(logger.to_dataframe())

    joined = (
        logger.to_dataframe()
        .set_index("run_id")
        .join(df.set_index("run_id")[["left_out", "model_type"]], on="run_id")
    )

    betas_df = (
        joined.groupby(["run_id", "model_type", "left_out", "Beta"])
        .apply(lambda x: roc_auc_score(x.IsOOD, -x.Score))
        .reset_index(["run_id", "model_type", "left_out", "Beta"])
    )
    betas_df.rename({0: "auroc"}, inplace=True, axis="columns")
    return betas_df


def make_dmd_comparison_figure():
    data = pd.read_csv(Path(DERIVATIVES_DIR) / "dmd_summary_final.csv")
    data["left_out"] = data.left_out.str.split("_").str.get(0)
    data.rename(
        columns={"auroc": "AUROC", "model_type": "Model Type"}, inplace=True
    )
    data = data.drop_duplicates(subset=['AUROC', 'left_out'])
    selected_rows = data['left_out'].isin(condition)
    data = data[selected_rows]
    data = data[data['beta'].isin(beta_set) | (data['Model Type'] == 'flat')]
    category_order =['flat','hier']
    g = sns.catplot(
        x="Model Type",
        y="AUROC",
        col="left_out",
        col_wrap=4,
        kind="box",
        sharey=False,
        data=data,
        order = category_order
    )
    
    st.pyplot(g.set_titles("{col_name}").fig)
    if 10 in beta_set:
        g.savefig(Path(REPORT_DIR) / "dmd_summary.pdf")
    else:
        g.savefig(Path(REPORT_DIR) / "dmd_summary_low.pdf")
    return data

def make_odin_comparison_figure():
    data = pd.read_csv(Path(DERIVATIVES_DIR) / "odin_summary_final.csv")
    data["left_out"] = df.left_out.str.split("_").str.get(0)
    data.rename(
        columns={"auroc": "AUROC", "model_type": "Model Type"}, inplace=True
    )
    data = data.drop_duplicates(subset=['AUROC', 'left_out'])
    selected_rows = data['left_out'].isin(condition)
    data = data[selected_rows]
    data = data[data['beta'].isin(beta_set) | (data['Model Type'] == 'flat')]
    category_order =['flat','hier']
    g = sns.catplot(
        x="Model Type",
        y="AUROC",
        col="left_out",
        col_wrap=4,
        kind="box",
        sharey=False,
        data = data,
        order = category_order,

    )
    st.pyplot(g.set_titles("{col_name}").fig)
    if 10 in beta_set:
        g.set_titles("{col_name}").savefig(
            Path(REPORT_DIR) / "odin_summary.pdf"
        )
    else:
        g.set_titles("{col_name}").savefig(
            Path(REPORT_DIR) / "odin_summary_low.pdf"
        )

    return data

def all_comparison(dmd_df,odin_df,msp_df):
    dmd_df.replace(
        {"Model Type": {"hier": "h_dmd", "flat": "f_dmd"}}, inplace=True
    )
    odin_df.replace(
        {"Model Type": {"hier": "h_odin", "flat": "f_odin"}},
        inplace=True,
    )
    msp_df.replace(
        {"Model Type": {"hier": "h_msp", "flat": "f_msp"}},
        inplace=True,
    )
    df_all =pd.concat([dmd_df, odin_df, msp_df])
    df_sorted = df_all.sort_values(['left_out'])
    category_order =["f_msp","h_msp","f_odin","h_odin","f_dmd" ,"h_dmd"]
    sns.set(font_scale=1.3)
    sns.set_style("white", {"axes.facecolor": "none"})
    g = sns.catplot(
        x="Model Type",
        y="AUROC",
        col="left_out",   
        col_wrap=4,
        kind="box",
        sharey=False,
        data=df_sorted,
        order = category_order,

    )
    st.pyplot(g.fig)
    if 10 in beta_set:
        g.savefig(Path(REPORT_DIR) / "all_comparison.pdf")
    else:
        g.savefig(Path(REPORT_DIR) / "all_comparison_low.pdf")


def sensitivity(dmd_df,odin_df,msp_df):
    
    dmd_df.replace(
        {"Model Type": {"hier": "h_dmd", "flat": "f_dmd"}}, inplace=True
    )
    odin_df.replace(
        {"Model Type": {"hier": "h_odin", "flat": "f_odin"}},
        inplace=True,
    )
    msp_df.replace(
        {"Model Type": {"hier": "h_msp", "flat": "f_msp"}},
        inplace=True,
    )
    df_all =pd.concat([dmd_df, odin_df, msp_df])
    df_all.drop()
    df_sorted = df_all.sort_values(['left_out'])

    category_order =["h_msp","h_odin" ,"h_dmd"]
    g = sns.catplot(
        x="Model Type",
        y="AUROC",
        col="left_out",
        col_wrap=4,
        kind="box",
        sharey=False,
        data=df_sorted,
        order = category_order,

        # data=pd.concat([dmd, odin, df]),
    )
    st.pyplot(g.fig)
    if 10 in beta_set:
        g.savefig(Path(REPORT_DIR) / "all_comparison.pdf")
    else:
        g.savefig(Path(REPORT_DIR) / "all_comparison_low.pdf")


if __name__ == "__main__":
    set_paper_style()

    df = pd.read_csv(Path(DERIVATIVES_DIR) / "msp_summary_final.csv")
    st.write(df.sort_values(by=["left_out", "auroc"], ascending=False))

    df_copy = df.copy()
    df_copy.rename(
        columns={"auroc": "AUROC", "model_type": "Model Type"}, inplace=True
    )
    df_copy["left_out"] = df_copy.left_out.str.split("_").str.get(0)
    df_msp = df_copy.drop_duplicates(subset=['AUROC', 'left_out'])

    selected_rows = df_msp['left_out'].isin(condition)
    df_msp = df_msp[selected_rows]
    df_msp = df_msp[df_msp['beta'].isin(beta_set) | (df_msp['Model Type'] == 'flat')]
    
    st.subheader("MSP")
    category_order =['flat','hier']
    g = sns.catplot(
        x="Model Type",
        y="AUROC",
        col="left_out",
        col_wrap=4,
        kind="box",
        sharey=False,
        data=df_msp,
        # data=df_copy,
        order = category_order
    )
    st.pyplot(g.set_titles("{col_name}"))



    st.subheader("Deep Mahalanobis Based Detection")
    dmd_df = make_dmd_comparison_figure()

    st.subheader("ODIN")
    odin_df = make_odin_comparison_figure()
    st.subheader("All Comparison")
    all_comparison(dmd_df.copy(),odin_df.copy(),df_msp.copy())
 