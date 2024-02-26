from anytree import Node
from anytree.search import find_by_attr
from anytree.util import commonancestors
from sklearn.preprocessing import LabelEncoder
import numpy as np


def make_tree_from(tree_descriptor: dict) -> Node:
    # TODO: this assumes a tree of only two levels, consider using anytree importers in the future
    root = Node("Root")
    for key, children in tree_descriptor.items():
        parent = Node(key, parent=root)
        [Node(child_key, parent=parent) for child_key in children]
    return root


def remove_from_tree(tree: Node, to_remove_name: str):
    parent_of_node_to_remove = find_by_attr(tree, to_remove_name).parent
    parent_of_node_to_remove.children = [
        child
        for child in parent_of_node_to_remove.children
        if child.name != to_remove_name
    ]
    # Clean up leaves that are not in the last level
    leaf_depth = max((node.depth for node in tree.leaves))
    for node in tree.leaves:
        if node.depth < leaf_depth:
            node.parent.children = [
                child
                for child in node.parent.children
                if child.name != node.name
            ]
    return tree


def lca_dist(tree: Node, node1_name: str, node2_name: str) -> float:
    if node1_name == node2_name:
        return 0.0
    tree_height = tree.height
    node1, node2 = [
        find_by_attr(tree, name) for name in [node1_name, node2_name]
    ]
    lca_height = commonancestors(node1, node2)[-1].height
    return lca_height / tree_height


def label_encoder_from(tree: Node) -> LabelEncoder:
    le = LabelEncoder()
    le.fit([leaf.name for leaf in tree.leaves])
    return le


class SoftlabelTransform(object):
    def __init__(self, tree: Node, beta: float) -> None:
        self.le = label_encoder_from(tree)
        num_classes_ = len(self.le.classes_)
        dist_matrix = np.empty((num_classes_, num_classes_), dtype=np.float)
        for i in range(num_classes_):
            for j in range(num_classes_):
                node_name_i, node_name_j = self.le.inverse_transform([i, j])
                dist_matrix[i, j] = lca_dist(tree, node_name_i, node_name_j)
        x = beta * np.negative(dist_matrix)
        e_x = np.exp(x - np.max(x))
        self.softmax = e_x / e_x.sum(axis=0) #shape (13,13)
    def __call__(self, label: str) -> np.ndarray:
        idx = self.le.transform([label])[0]
        return self.softmax[idx, :]  #shape (13,)


class HardLabelTransform(object):
    def __init__(self, tree: Node) -> None:
        self.le = label_encoder_from(tree)

    def __call__(self, label: str):
        return self.le.transform([label])[0]