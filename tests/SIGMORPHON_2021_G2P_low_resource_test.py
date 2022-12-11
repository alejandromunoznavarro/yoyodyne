"""Tests the basic models on a g2p task.

We use the low-resource subtask of the SIGMORPHON 2021 shared task on
grapheme-to-phoneme conversion, in which participants are provided with 800
training examples (and 100 dev and test examples, respectively) for each of 10
languages. Naturally this is something of a "change-detector" test."""

import pathlib

from typing import Tuple

import pytest

from yoyodyne import train, predict

# TODO(kbg): This assumes an API we do not have yet. Add it.
# TODO(kbg0: Add the actual numbers in once we have them.
# TODO(kbg): Can we use GPU instances on CircleCI for this?


# Computes the relevant testdata directory.
TESTDATA = (
    pathlib.Path(__file__).parent
    / "testdata"
    / "SIGMORPHON_2021_G2P_low_resource"
)


def get_data_paths(language: str) -> Tuple[str, str, str]:
    train_path = str(TESTDATA / language / "train.tsv")
    dev_path = str(TESTDATA / language / "dev.tsv")
    test_path = str(TESTDATA / language / "test.tsv")
    return train_path, dev_path, test_path


@pytest.mark.parametrize(
    "language, expected_accuracy",
    [
        ("ady", 1.0),
        ("gre", 1.0),
        ("ice", 1.0),
        ("ita", 1.0),
        ("khm", 1.0),
        ("lav", 1.0),
        ("mlt_latn", 1.0),
        ("rum", 1.0),
        ("slv", 1.0),
        ("wel_sw", 1.0),
    ],
)
def test_bilstm(language, expected_accuracy):
    train_path, dev_path, test_path = get_data_paths(language)
    best_model = train.train(train_path, dev_path, **kwargs)
    predict.accuracy(best_model, test_path)
    assert accuracy == pytest.approx(expected_accuracy, rel=0.001)


@pytest.mark.parametrize(
    "language, accuracy",
    [
        ("ady", 1.0),
        ("gre", 1.0),
        ("ice", 1.0),
        ("ita", 1.0),
        ("khm", 1.0),
        ("lav", 1.0),
        ("mlt_latn", 1.0),
        ("rum", 1.0),
        ("slv", 1.0),
        ("wel_sw", 1.0),
    ],
)
def test_bilstm_attention(language, accuracy):
    train_path, dev_path, test_path = get_data_paths(language)
    best_model = train.train(train_path, dev_path, **kwargs)
    predict.accuracy(best_model, test_path)
    assert accuracy == pytest.approx(expected_accuracy, rel=0.001)


@pytest.mark.parametrize(
    "language, accuracy",
    [
        ("ady", 1.0),
        ("gre", 1.0),
        ("ice", 1.0),
        ("ita", 1.0),
        ("khm", 1.0),
        ("lav", 1.0),
        ("mlt_latn", 1.0),
        ("rum", 1.0),
        ("slv", 1.0),
        ("wel_sw", 1.0),
    ],
)
def test_pointer_generator_lstm(language, accuracy):
    train_path, dev_path, test_path = get_data_paths(language)
    best_model = train.train(train_path, dev_path, **kwargs)
    predict.accuracy(best_model, test_path)
    assert accuracy == pytest.approx(expected_accuracy, rel=0.001)


@pytest.mark.parametrize(
    "language, accuracy",
    [
        ("ady", 1.0),
        ("gre", 1.0),
        ("ice", 1.0),
        ("ita", 1.0),
        ("khm", 1.0),
        ("lav", 1.0),
        ("mlt_latn", 1.0),
        ("rum", 1.0),
        ("slv", 1.0),
        ("wel_sw", 1.0),
    ],
)
def test_transducer(language, accuracy):
    train_path, dev_path, test_path = get_data_paths(language)
    best_model = train.train(train_path, dev_path, **kwargs)
    predict.accuracy(best_model, test_path)
    assert accuracy == pytest.approx(expected_accuracy, rel=0.001)


@pytest.mark.parametrize(
    "language, accuracy",
    [
        ("ady", 1.0),
        ("gre", 1.0),
        ("ice", 1.0),
        ("ita", 1.0),
        ("khm", 1.0),
        ("lav", 1.0),
        ("mlt_latn", 1.0),
        ("rum", 1.0),
        ("slv", 1.0),
        ("wel_sw", 1.0),
    ],
)
def test_tranformer(language, accuracy):
    train_path, dev_path, test_path = get_data_paths(language)
    best_model = train.train(train_path, dev_path, **kwargs)
    predict.accuracy(best_model, test_path)
    assert accuracy == pytest.approx(expected_accuracy, rel=0.001)


@pytest.mark.parametrize(
    "language, accuracy",
    [
        ("ady", 1.0),
        ("gre", 1.0),
        ("ice", 1.0),
        ("ita", 1.0),
        ("khm", 1.0),
        ("lav", 1.0),
        ("mlt_latn", 1.0),
        ("rum", 1.0),
        ("slv", 1.0),
        ("wel_sw", 1.0),
    ],
)
def test_unilstm_attention(language, accuracy):
    train_path, dev_path, test_path = get_data_paths(language)
    best_model = train.train(train_path, dev_path, **kwargs)
    predict.accuracy(best_model, test_path)
    assert accuracy == pytest.approx(expected_accuracy, rel=0.001)
