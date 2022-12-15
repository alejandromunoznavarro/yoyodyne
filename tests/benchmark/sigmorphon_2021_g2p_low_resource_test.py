"""Tests the basic models on a g2p task.

We use the low-resource subtask of the SIGMORPHON 2021 shared task on
grapheme-to-phoneme conversion, in which participants are provided with 800
training examples (and 100 dev and test examples, respectively) for each of 10
languages. Naturally this is something of a "change-detector" test."""

import pathlib

from typing import Tuple

import pytest

from yoyodyne import train, predict

# TODO(kbg): Add the actual numbers in once they are available.
# TODO(kbg): Can we use GPU instances on CircleCI?
# TODO(kbg): Shard this.

# Computes the relevant testdata directory.
TESTDATA = (
    pathlib.Path(__file__).parent
    / "testdata"
    / "sigmorphon_2021_g2p_low_resource"
)

# Numerical tolerance for accuracy; there are 100 test examples.
REL = 0.01

# Arguments shared across all experiments.
SHARED = {
    "seed": 49,
    "features_col": 0,  # No features are present here.
    "target_col": 2,  # Needed for accuracy calculation.
    "gradient_clip": 3,
    "gpu": False,
}


def get_data_paths(language: str) -> Tuple[str, str, str]:
    """Helper for computing the three input file paths."""
    train_path = str(TESTDATA / language / "train.tsv")
    dev_path = str(TESTDATA / language / "dev.tsv")
    test_path = str(TESTDATA / language / "test.tsv")
    return train_path, dev_path, test_path


@pytest.fixture
def make_model_dir(tmp_path):
    """Creates a temporary model directory."""
    path = tmp_path / "models"
    path.mkdir()
    return path


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
def test_bilstm(make_model_dir, language, expected_accuracy):
    train_path, dev_path, test_path = get_data_paths(language)
    best_model = train.train(
        language,
        train_path,
        dev_path,
        make_model_dir,
        arch="lstm",
        attention=False,
        tied_vocabulary=False,
        **SHARED,
    )
    predictor = predict.Predictor(
        language,
        test_path,
        make_model_dir,
        best_model,
        arch="lstm",
        attention=False,
        tied_vocabulary=False,
        **SHARED,
    )
    accuracy = predictor.accuracy()
    print(f"Language:\t{language};\tAccuracy: {accuracy:.4f}")
    # FIXME: bring assertion back.


"""
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
def test_bilstm_attention(make_model_dir, language, expected_accuracy):
    train_path, dev_path, test_path = get_data_paths(language)
    best_model = train.train(
        language,
        train_path,
        dev_path,
        make_model_dir,
        arch="lstm",
        tied_vocabulary=False,
        **SHARED,
    )
    predictor = predict.Predictor(
        language,
        test_path,
        make_model_dir,
        best_model,
        arch="lstm",
        tied_vocabulary=False,
        **SHARED,
    )
    assert predictor.accuracy() == pytest.approx(expected_accuracy, rel=REL)


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
def test_pointer_generator_lstm(make_model_dir, language, expected_accuracy):
    train_path, dev_path, test_path = get_data_paths(language)
    best_model = train.train(
        language,
        train_path,
        dev_path,
        make_model_dir,
        arch="pointer_generator_lstm",
        **SHARED,
    )
    predictor = predict.Predictor(
        language,
        test_path,
        make_model_dir,
        best_model,
        arch="pointer_generator_lstm",
        **SHARED,
    )
    assert predictor.accuracy() == pytest.approx(expected_accuracy, rel=REL)


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
def test_transducer(make_model_dir, language, expected_accuracy):
    train_path, dev_path, test_path = get_data_paths(language)
    best_model = train.train(
        language,
        train_path,
        dev_path,
        make_model_dir,
        arch="transducer",
        **SHARED,
    )
    predictor = predict.Predictor(
        language,
        test_path,
        make_model_dir,
        best_model,
        arch="pointer_generator_lstm",
        **SHARED,
    )
    assert predictor.accuracy() == pytest.approx(expected_accuracy, rel=REL)


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
def test_tranformer(make_model_dir, language, expected_accuracy):
    train_path, dev_path, test_path = get_data_paths(language)
    best_model = train.train(
        language,
        train_path,
        dev_path,
        make_model_dir,
        tied_vocabulary=False,
        arch="transformer",
        eval_every=8,
        max_epochs=400,
        beta2=0.98,
        **SHARED,
    )
    predictor = predict.Predictor(
        language,
        test_path,
        make_model_dir,
        best_model,
        arch="transformer",
        **SHARED,
    )
    assert predictor.accuracy() == pytest.approx(expected_accuracy, rel=REL)


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
def test_unilstm_attention(make_model_dir, language, expected_accuracy):
    train_path, dev_path, test_path = get_data_paths(language)
    best_model = train.train(
        language,
        train_path,
        dev_path,
        make_model_dir,
        tied_vocabulary=False,
        bidirectional=False,
        eval_every=8,
        max_epochs=400,
        beta2=0.98,
        **SHARED,
    )
    predictor = predict.Predictor(
        language,
        test_path,
        make_model_dir,
        model_path,
        bidirectional=False,
        **SHARED,
    )
    assert predictor.accuracy() == pytest.approx(expected_accuracy, rel=REL)
"""
