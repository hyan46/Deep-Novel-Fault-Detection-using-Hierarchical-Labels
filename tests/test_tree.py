import pytest
from anytree import LevelGroupOrderIter
from anytree.search import find_by_attr
from numpy.testing import assert_equal
from src.tree import (
    make_tree_from,
    remove_from_tree,
    lca_dist,
    label_encoder_from,
)
from src.utils import load_json
from settings import PROJECT_HOME


@pytest.fixture
def tree_decriptor_json():
    return load_json(PROJECT_HOME / "tests" / "tree_description_fixture.json")


def test_make_tree_from_levels(tree_decriptor_json):
    tree = make_tree_from(tree_decriptor_json)
    root_level, level_1, level_2 = LevelGroupOrderIter(tree)
    assert {node.name for node in root_level} == {"Root"}
    assert {node.name for node in level_1} == {"A", "B", "C"}
    assert {node.name for node in level_2} == {"0", "1", "2", "3", "4", "5"}


def test_make_tree_from_children(tree_decriptor_json):
    tree = make_tree_from(tree_decriptor_json)
    assert {node.name for node in find_by_attr(tree, "A").children} == {
        "0",
        "1",
        "2",
    }

    assert {node.name for node in find_by_attr(tree, "B").children} == {
        "3",
        "4",
    }

    assert {node.name for node in find_by_attr(tree, "C").children} == {
        "5",
    }


def test_remove_from_tree(tree_decriptor_json):
    tree = make_tree_from(tree_decriptor_json)
    tree = remove_from_tree(tree, "0")
    node_A = find_by_attr(tree, "A")
    assert {node.name for node in node_A.children} == {"1", "2"}


@pytest.mark.parametrize(
    "node_to_delete, parent, expected", [("5", "C", False), ("3", "B", False)]
)
def test_remove_from_tree_childless_parents(
    tree_decriptor_json, node_to_delete, parent, expected
):
    tree = make_tree_from(tree_decriptor_json)
    tree = remove_from_tree(tree, node_to_delete)
    leaves = {node.name for node in tree.leaves}
    assert (parent in leaves) == expected


def test_lca_dist(tree_decriptor_json):
    tree = make_tree_from(tree_decriptor_json)
    assert lca_dist(tree, "0", "0") == 0.0  # lowest common ancestor is itself
    assert lca_dist(tree, "0", "1") == 0.5  # lowest common ancestor is 'A'
    assert lca_dist(tree, "0", "3") == 1.0  # lowest common ancestor is root


def test_label_encder_from_tree(tree_decriptor_json):
    tree = make_tree_from(tree_decriptor_json)
    le = label_encoder_from(tree)
    assert le.transform(["0"]) == [0]
    assert le.transform(["3"]) == [3]
    assert le.transform(["5"]) == [5]
