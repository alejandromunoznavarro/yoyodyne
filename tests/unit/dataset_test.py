import csv
import pytest

from yoyodyne import datasets


@pytest.fixture
def make_tsv_file(tmp_path):
    path = tmp_path / "data.tsv"
    with open(path, "w") as sink:
        tsv_writer = csv.writer(sink, delimiter="\t")
        tsv_writer.writerow(["abscond", "absconded", "regular"])
        tsv_writer.writerow(["muff", "muffed", "regular"])
        tsv_writer.writerow(["outshine", "outshone", "irregular"])
    return path


@pytest.mark.parametrize(
    "features_col, expected_cls",
    [(0, datasets.DatasetNoFeatures), (3, datasets.DatasetFeatures)],
)
def test_get_dataset(make_tsv_file, features_col, expected_cls):
    filename = make_tsv_file
    dataset = datasets.get_dataset(
        filename,
        tied_vocabulary=True,  # Doesn't matter.
        source_col=1,  # Doesn't matter.
        target_col=2,  # Doesn't matter.
        features_col=features_col,
    )
    assert type(dataset) is expected_cls
