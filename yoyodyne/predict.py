"""Prediction."""

import csv
import os

from typing import Iterator, Optional

import click
import pytorch_lightning as pl
import torch
from torch.utils import data

from . import collators, datasets, models, util


class Predictor:
    """Data for making predictions."""

    source_col: int
    target_col: int
    features_col: int
    source_sep: str
    target_sep: str
    features_sep: str
    batch_size: int
    beam_width: Optional[int]
    dataset: data.Dataset
    collator = collators.Collator
    loader: data.DataLoader
    model: pl.LightningModule
    trainer: pl.Trainer

    def __init__(
        self,
        experiment: str,
        # Path arguments.
        input_path: str,
        model_dir: str,
        model_path: str,
        *,
        # Data format arguments.
        source_col=1,
        target_col=0,
        features_col=0,
        source_sep="",
        target_sep="",
        features_sep=";",
        # Architecture arguments.
        arch: str = "lstm",
        attention: bool = True,
        tied_vocabulary: bool = True,
        # Prediction arguments.
        batch_size: int = 128,
        beam_width: Optional[int] = None,
        gpu: bool = True,
    ):
        self.source_col = source_col
        self.target_col = target_col
        self.features_col = features_col
        self.source_sep = source_sep
        self.target_sep = target_sep
        self.features_sep = features_sep
        self.arch = arch
        self.batch_size = batch_size
        self.beam_width = beam_width
        if self.beam_width is not None:
            util.log_info("Decoding with beam search; forcing batch size to 1")
            self.batch_size = 1
        # Creates dataset.
        self.dataset = datasets.get_dataset(
            input_path,
            source_col=source_col,
            target_col=target_col,
            features_col=features_col,
            source_sep=source_sep,
            target_sep=target_sep,
            features_sep=features_sep,
            tied_vocabulary=tied_vocabulary,
        )
        self.dataset.load_index(model_dir, experiment)
        # Creates loader.
        self.include_features = features_col != 0
        self.include_target = target_col != 0
        self.collator = collators.get_collator(
            self.dataset.pad_idx,
            arch=arch,
            include_features=self.include_features,
            include_target=self.include_target,
        )
        self.loader = data.DataLoader(
            self.dataset,
            collate_fn=self.collator,
            batch_size=batch_size,
            shuffle=False,
        )
        # Creates "trainer".
        model_cls = models.get_model_cls(
            arch, attention, self.include_features
        )
        util.log_info(f"Model: {model_cls.__name__}")
        device = util.get_device(gpu)
        self.model = model_cls.load_from_checkpoint(model_path).to(device)
        self.model.beam_width = self.beam_width
        self.model.eval()
        self.trainer = pl.Trainer(
            accelerator="gpu" if gpu and torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=0,  # Silences a warning.
            num_sanity_val_steps=0,
        )

    # TODO(kbg): Add class method to avoid some of this work for writing dev
    # predictions.

    def source(self) -> Iterator[str]:
        """Generates all source strings.

        Yields:
            (str) sources.
        """
        for batch in self.loader:
            source_lists = self.dataset.decode_source(
                batch[0], symbols=True, special=False
            )
            for source_list in source_lists:
                yield self.source_sep.join(source_list)

    def features(self) -> Iterator[str]:
        """Generates all target strings.

        This is only valid if features_col > 0.

        Yields:
            (str) features.
        """
        assert self.include_features
        for batch in self.loader:
            features_lists = self.dataset.decode_features(
                batch[2] if self.collator.has_features else batch[0],
                symbols=True,
                special=False,
            )
            for features_list in features_lists:
                yield self.target_sep.join(features_list)

    def target(self) -> Iterator[str]:
        """Generates all target strings.

        This is only valid if target_col > 0.

        Yields:
            (str) targets.
        """
        assert self.include_target
        for batch in self.loader:
            target_lists = self.dataset.decode_target(
                batch[-2], symbols=True, special=False
            )
            for target_list in target_lists:
                yield self.target_sep.join(target_list)

    def prediction(self) -> Iterator[str]:
        """Generates all final predictions.

        Yields:
            (str) predictions.
        """
        util.log_info("Predicting...")
        for batch in self.trainer.predict(self.model, dataloaders=self.loader):
            if self.arch not in ["transducer"]:
                # -> B x seq_len x vocab_size
                batch = batch.transpose(1, 2)
                if self.beam_width is not None:
                    batch = batch.squeeze(2)
                else:
                    _, batch = torch.max(batch, dim=2)
            # Uses CPU because PL seems to always return CPU tensors.
            batch = self.model.evaluator.finalize_preds(
                batch, self.dataset.end_idx, self.dataset.pad_idx, "cpu"
            )
            for prediction in batch:
                for prediction_list in self.dataset.decode_target(
                    batch, symbols=True, special=False
                ):
                    yield self.target_sep.join(prediction_list)
        util.log_info("Prediction complete")

    def accuracy(self) -> float:
        """Computes accuracy.

        This is only valid if target_col > 0.

        Returns:
            (float) accuracy.
        """
        correct = 0
        total = 0
        for gold, predicted in zip(self.target(), self.prediction()):
            if gold == predicted:
                correct += 1
            total += 1
        return correct / total

    def write(self, output_path: str) -> None:
        """Writes predictions to TSV file.

        * The source string is always written.
        * The prediction is always written.
        * The features are written if features_col > 0.
        * The target string is never written.

        Args:
           output_path (str).
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        util.log_info(f"Writing predictions to {output_path}")
        with open(output_path, "w") as sink:
            tsv_writer = csv.writer(sink, delimiter="\t")
            row_template = [""] * max(
                self.source_col, self.target_col, self.features_col
            )
            if self.include_features:
                for source, features, prediction in zip(
                    self.source(), self.features(), self.prediction()
                ):
                    row = row_template.copy()
                    row[self.source_col - 1] = source
                    row[self.target_col - 1] = prediction
                    row[self.features_col - 1] = features
                    tsv_writer.writerow(row)
            else:
                for source, prediction in zip(
                    self.source(), self.prediction()
                ):
                    row = row_template.copy()
                    row[self.source_col - 1] = source
                    row[self.target_col - 1] = prediction
                    tsv_writer.writerow(row)


@click.command()
@click.option(
    "--experiment", type=str, required=True, help="Name of experiment"
)
@click.option(
    "--input-path", type=str, required=True, help="Path to inputs to predict"
)
@click.option(
    "--model-dir",
    type=str,
    required=True,
    help="Path to output directory for models",
)
@click.option("--model-path", type=str, required=True, help="Path to model")
@click.option(
    "--output-path", type=str, required=True, help="Path for predicted outputs"
)
@click.option(
    "--source-col",
    type=int,
    default=1,
    help="1-based index for the source column",
)
@click.option(
    "--target-col",
    type=int,
    default=2,
    help="1-based index for the target column",
)
@click.option(
    "--features-col",
    type=int,
    default=0,
    help="1-based index for the features column; "
    "0 (the default) indicates the model will not use features",
)
@click.option(
    "--source-sep",
    type=str,
    default="",
    help="String used to split source string into symbols; "
    "an empty string (the default) indicates that each Unicode codepoint in "
    "the source string is its own symbol",
)
@click.option(
    "--target-sep",
    type=str,
    default="",
    help="String used to split target string into symbols; "
    "an empty string (the default) indicates that each Unicode codepoint in "
    "the target string is its own symbol",
)
@click.option(
    "--features-sep",
    type=str,
    default=";",
    help="String used to split the features string into symbols; "
    "an empty string indicates that each unicode codepoint in the features "
    "string is its own symbol",
)
@click.option(
    "--arch",
    type=click.Choice(
        [
            "feature_invariant_transformer",
            "lstm",
            "pointer_generator_lstm",
            "transducer",
            "transformer",
        ]
    ),
    default="lstm",
    help="Model architecture to use",
)
@click.option(
    "--attention/--no-attention",
    type=bool,
    default=True,
    help="Uses attention (LSTM architecture only; ignored otherwise)",
)
@click.option(
    "--tied-vocabulary/--no-tied-vocabulary",
    default=True,
    help="Share embeddings between source and target",
)
@click.option(
    "--batch-size", type=int, default=128, help="Batch size for prediction"
)
@click.option(
    "--beam-width",
    type=int,
    help="Optional: width for beam search; "
    "if not specified beam search is not used",
)
def main(
    experiment,
    # Path arguments.
    input_path,
    model_dir,
    model_path,
    output_path,
    # Data format arguments.
    source_col,
    target_col,
    features_col,
    source_sep,
    target_sep,
    features_sep,
    # Architecture arguments.
    arch,
    attention,
    tied_vocabulary,
    # Prediction arguments.
    batch_size,
    beam_width,
):
    """Makes predictions with a trained sequence-to-sequence model.

    Args:
        experiment (str).
        input_path (str).
        model_path (str).
        output_path (str).
        source_col (int).
        target_col (int).
        features_col (int).
        source_sep (str).
        target_sep (str).
        features_sep (str).
        arch (str).
        attention (str).
        tied_vocabulary (bool).
        batch_size (int).
        beam_width (int, optional).
    """
    util.log_info("Arguments:")
    for arg, val in click.get_current_context().params.items():
        util.log_info(f"\t{arg}: {val!r}")
    predictor = Predictor(
        experiment,
        input_path,
        model_dir,
        model_path,
        source_col=source_col,
        target_col=target_col,
        features_col=features_col,
        source_sep=source_sep,
        features_sep=features_sep,
        arch=arch,
        attention=attention,
        tied_vocabulary=tied_vocabulary,
        batch_size=batch_size,
        beam_width=beam_width,
    )
    predictor.write(output_path)


if __name__ == "__main__":
    main()
