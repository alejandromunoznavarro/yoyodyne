"""Training."""

import os
import time
from typing import Optional

import click
import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks, loggers
from torch.utils import data

from . import collators, datasets, evaluators, models, predict, util


def train(
    experiment: str,
    # Path arguments.
    train_path: str,
    dev_path: str,
    model_dir: str,
    *,
    train_from_path: Optional[str] = None,
    # Data format arguments.
    source_col: int = 1,
    target_col: int = 2,
    features_col: int = 0,
    source_sep: str = "",
    target_sep: str = "",
    features_sep: str = "",
    # Architecture arguments.
    arch: str = "lstm",
    attention: bool = True,
    attention_heads: int = 4,
    bidirectional: bool = True,
    decoder_layers: int = 1,
    embedding_size: int = 128,
    encoder_layers: int = 1,
    hidden_size: int = 512,
    max_sequence_length: int = 128,
    max_decode_length: int = 128,
    oracle_em_epochs: int = 5,
    oracle_factor: int = 1,
    sed_path: Optional[str] = None,
    tied_vocabulary: bool = True,
    # Training arguments.
    batch_size: int = 128,
    beta1: float = 0.9,
    beta2: float = 0.999,
    dataloader_workers: int = 1,
    dropout: float = 0.2,
    max_epochs: int = 40,
    eval_batch_size: int = 128,
    eval_every: int = 2,
    gradient_clip: Optional[float] = None,
    gpu: bool = True,
    label_smoothing: Optional[float] = None,
    learning_rate: float = 0.001,
    patience: Optional[int] = 0,
    optimizer: str = "adam",
    save_top_k: int = 1,
    scheduler: str = "",
    seed: int = time.time_ns(),
    warmup_steps: int = 0,
    wandb: bool = False,
    # Development predictions.
    dev_predictions_path: Optional[str] = None,
) -> str:
    """Performs training, returning the path to the best model.

    Args:
        experiment (str).
        train_path (str).
        dev_path (str).
        model_dir (str).
        train_from_path (str, optional).
        source_col (int).
        target_col (int).
        features_col (int).
        source_sep (str).
        target_sep (str).
        features_sep (str).
        arch (str).
        attention (bool).
        attention_heads (int).
        bidirectional (bool).
        decoder_layers (int).
        embedding_size (int).
        encoder_layers (int).
        hidden_size (int).
        max_sequence_length (int).
        max_decode_length (int).
        oracle_em_epochs (int).
        oracle_factor (int).
        sed_path (str, optional).
        tied_vocabulary (bool).
        batch_size (int).
        beta1 (float).
        beta2 (float).
        dataloader_workers (int)
        dropout (float).
        max_epochs (int).
        eval_batch_size (int).
        eval_every (int).
        gradient_clip (float, optional).
        gpu (bool).
        label_smoothing (float, optional).
        learning_rate (float)
        patience (int, optional).
        optimizer (str)
        save_top_k (int).
        scheduler (string).
        seed (int).
        warmup_steps (int).
        wandb (bool).
        dev_predictions_path (str, optional)

    Returns:
        The path to the best model.
    """
    util.seed(seed)
    # Sets up data sets, collators, and loaders.
    if target_col == 0:
        raise datasets.Error("target_col must be specified for training")
    include_features = features_col != 0
    train_set = datasets.get_dataset(
        train_path,
        tied_vocabulary=tied_vocabulary,
        source_col=source_col,
        target_col=target_col,
        features_col=features_col,
        source_sep=source_sep,
        target_sep=target_sep,
        features_sep=features_sep,
    )
    util.log_info(f"Source vocabulary: {train_set.source_symbol2i}")
    util.log_info(f"Target vocabulary: {train_set.target_symbol2i}")
    collator = collators.get_collator(
        train_set.pad_idx,
        arch=arch,
        include_features=include_features,
        include_target=True,
    )
    train_loader = data.DataLoader(
        train_set,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
    )
    dev_set = datasets.get_dataset(
        dev_path,
        tied_vocabulary=tied_vocabulary,
        source_col=source_col,
        target_col=target_col,
        features_col=features_col,
        source_sep=source_sep,
        target_sep=target_sep,
        features_sep=features_sep,
    )
    dev_loader = data.DataLoader(
        dev_set,
        collate_fn=collator,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
    )
    # Sets up evaluator.
    device = util.get_device(gpu)
    evaluator = evaluators.Evaluator(device=device)
    # Sets up logger and trainer.
    logger = [loggers.CSVLogger(model_dir, name=experiment)]
    if wandb:
        logger.append(loggers.WandbLogger(project=experiment, log_model="all"))
    # ckp_callback is used later for logging the best checkpoint path.
    ckp_callback = callbacks.ModelCheckpoint(
        save_top_k=save_top_k,
        monitor="val_accuracy",
        mode="max",
        filename="model-{epoch:02d}-{val_accuracy:.2f}",
    )
    trainer_callbacks = []
    if patience is not None:
        trainer_callbacks.append(
            callbacks.early_stopping.EarlyStopping(
                monitor="val_accuracy",
                min_delta=0.00,
                patience=patience,
                verbose=False,
                mode="max",
            )
        )
    trainer_callbacks.append(ckp_callback)
    trainer_callbacks.append(
        callbacks.LearningRateMonitor(logging_interval="epoch")
    )
    trainer_callbacks.append(callbacks.TQDMProgressBar())
    trainer = pl.Trainer(
        accelerator="gpu" if gpu and torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        max_epochs=max_epochs,
        gradient_clip_val=gradient_clip,
        check_val_every_n_epoch=eval_every,
        enable_checkpointing=True,
        default_root_dir=model_dir,
        callbacks=trainer_callbacks,
        log_every_n_steps=len(train_set) // batch_size,
        num_sanity_val_steps=0,
    )
    # So we can write indices to it before PL creates it.
    os.makedirs(trainer.loggers[0].log_dir, exist_ok=True)
    train_set.write_index(trainer.loggers[0].log_dir, experiment)
    dev_set.load_index(trainer.loggers[0].log_dir, experiment)
    # Trains model.
    model_cls = models.get_model_cls(arch, attention, include_features)
    util.log_info(f"Model: {model_cls.__name__}")
    if train_from_path is not None:
        util.log_info(f"Loading model from {train_from_path}")
        model = model_cls.load_from_checkpoint(train_from_path).to(device)
        util.log_info("Training...")
        trainer.fit(model, train_loader, dev_loader, ckpt_path=train_from_path)
    else:
        expert = None
        if arch in ["transducer"]:
            expert = models.expert.get_expert(
                train_set,
                epochs=oracle_em_epochs,
                oracle_factor=oracle_factor,
                sed_path=sed_path,
            )
        model = model_cls(
            beta1=beta1,
            beta2=beta2,
            bidirectional=bidirectional,
            decoder_layers=decoder_layers,
            dropout=dropout,
            encoder_layers=encoder_layers,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            end_idx=train_set.end_idx,
            evaluator=evaluator,
            expert=expert,
            features_idx=getattr(train_set, "features_idx", -1),
            features_vocab_size=getattr(train_set, "features_vocab_size", -1),
            label_smoothing=label_smoothing,
            learning_rate=learning_rate,
            max_decode_length=max_decode_length,
            max_sequence_length=max_sequence_length,
            attention_heads=attention_heads,
            optimizer=optimizer,
            output_size=train_set.target_vocab_size,
            scheduler=scheduler,
            pad_idx=train_set.pad_idx,
            start_idx=train_set.start_idx,
            vocab_size=train_set.source_vocab_size,
            warmup_steps=warmup_steps,
        ).to(device)
        util.log_info("Training...")
        trainer.fit(model, train_loader, dev_loader)
    util.log_info("Training complete")
    best_model_path = ckp_callback.best_model_path
    # If specified, write dev set predictions.
    # TODO: Add beam-width option so we can make predictions with beam search.
    if dev_predictions_path:
        util.log_info(f"Writing dev predictions to {dev_predictions_path}")
        model = model_cls.load_from_checkpoint(best_model_path).to(device)
        predict.predict(
            model,
            dev_loader,
            dev_predictions_path,
            arch,
            batch_size,
            source_col,
            target_col,
            features_col,
            source_sep,
            target_sep,
            features_sep,
            include_features,
            gpu,
        )
    return best_model_path


@click.command()
@click.option(
    "--experiment", type=str, required=True, help="Name of experiment"
)
@click.option(
    "--train-path", type=str, required=True, help="Path to input training data"
)
@click.option(
    "--dev-path",
    type=str,
    required=True,
    help="Path to input development data",
)
@click.option(
    "--model-dir",
    type=str,
    required=True,
    help="Path to output directory for models",
)
@click.option(
    "--train-from-path",
    type=str,
    help="Optional: training will begin from this checkpoint",
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
    "--attention-heads",
    type=int,
    default=4,
    help="Number of attention heads "
    "(transformer-backed architectures only; ignored otherwise)",
)
@click.option(
    "--bidirectional/--no-bidirectional",
    type=bool,
    default=True,
    help="Uses a bidirectional encoder "
    "(LSTM-backed architectures only; ignored otherwise)",
)
@click.option(
    "--decoder-layers", type=int, default=1, help="Number of decoder layers"
)
@click.option(
    "--embedding-size",
    type=int,
    default=128,
    help="Dimensionality of embeddings",
)
@click.option(
    "--encoder-layers", type=int, default=1, help="Number of encoder layers"
)
@click.option(
    "--hidden-size",
    type=int,
    default=512,
    help="Dimensionality of the hidden layer(s)",
)
@click.option("--max-sequence-length", type=int, default=128)
@click.option("--max-decode-length", type=int, default=128)
@click.option(
    "--oracle-em-epochs",
    type=int,
    default=5,
    help="Number of EM epochs "
    "(transducer architecture only; ignored otherwise)",
)
@click.option(
    "--oracle-factor",
    type=int,
    default=1,
    help="Roll-in schedule parameter "
    "(transducer architecture only; ignored otherwise)",
)
@click.option(
    "--sed-path",
    type=str,
    help="Path to SED parameters "
    "(transducer architecture only; ignored otherwise)",
)
@click.option(
    "--tied-vocabulary/--no-tied-vocabulary",
    default=True,
    help="Share embeddings between source and target",
)
@click.option(
    "--batch-size", type=int, default=128, help="Batch size for training"
)
@click.option(
    "--beta1",
    type=float,
    default=0.9,
    help="beta_1 for Adam optimizer (ignored otherwise)",
)
@click.option(
    "--beta2",
    type=float,
    default=0.9,
    help="beta_2 for Adam optimizer (ignored otherwise)",
)
@click.option(
    "--dataloader-workers", type=int, default=1, help="Number of data loaders"
)
@click.option("--dropout", type=float, default=0.2, help="Dropout probability")
@click.option(
    "--eval-batch-size",
    type=int,
    default=128,
    help="Batch size for evaluation",
)
@click.option(
    "--eval-every", type=int, default=2, help="Number of epochs per evaluation"
)
@click.option(
    "--gradient-clip",
    type=float,
    help="Optional: threshold for gradient clipping",
)
@click.option("--gpu/--no-gpu", default=True, help="Use GPU")
@click.option(
    "--label-smoothing",
    type=float,
    help="Optional: coefficient for label smoothing",
)
@click.option(
    "--learning-rate", type=float, default=0.001, help="Learning rate"
)
@click.option(
    "--max-epochs", type=int, default=40, help="Maximum number of epochs"
)
@click.option(
    "--optimizer",
    type=click.Choice(["adadelta", "adam", "sgd"]),
    default="adam",
    help="Optimizer",
)
@click.option(
    "--patience",
    type=int,
    help="Optional: number of evaluations without any improvement needed to "
    "stop training early",
)
@click.option(
    "--save-top-k",
    type=int,
    default=1,
    help="Number of the best models to be saved",
)
@click.option(
    "--scheduler",
    type=click.Choice(["warmupinvsqrt", ""]),
    default="",
    help="Optional: learning rate scheduler",
)
@click.option(
    "--seed",
    type=int,
    default=time.time_ns(),
    help="Optional: random seed (current time used if not specified)",
)
@click.option(
    "--warmup-steps",
    type=int,
    default=0,
    help="Number of warmup steps "
    "(warmupinvsq scheduler only; ignored otherwise",
)
@click.option(
    "--wandb/--no-wandb",
    default=False,
    help="Use Weights & Biases logging (log-in required)",
)
@click.option(
    "--dev-predictions-path",
    type=str,
    help="Optional: path for best model's development set predictions",
)
def main(
    experiment,
    # Path arguments.
    train_path,
    dev_path,
    model_dir,
    train_from_path,
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
    attention_heads,
    bidirectional,
    decoder_layers,
    embedding_size,
    encoder_layers,
    hidden_size,
    max_decode_length,
    max_sequence_length,
    oracle_em_epochs,
    oracle_factor,
    sed_path,
    tied_vocabulary,
    # Training arguments.
    batch_size,
    beta1,
    beta2,
    dataloader_workers,
    dropout,
    eval_batch_size,
    eval_every,
    gradient_clip,
    gpu,
    label_smoothing,
    learning_rate,
    max_epochs,
    optimizer,
    patience,
    save_top_k,
    scheduler,
    seed,
    warmup_steps,
    wandb,
    # Development predictions.
    dev_predictions_path,
):
    """Trains a sequence-to-sequence model."""
    util.log_info("Arguments:")
    for arg, val in click.get_current_context().params.items():
        util.log_info(f"\t{arg}: {val!r}")
    best_model_path = train(
        experiment,
        train_path,
        dev_path,
        model_dir,
        train_from_path=train_from_path,
        source_col=source_col,
        target_col=target_col,
        features_col=features_col,
        source_sep=source_sep,
        target_sep=target_sep,
        features_sep=features_sep,
        arch=arch,
        attention=attention,
        attention_heads=attention_heads,
        bidirectional=bidirectional,
        embedding_size=embedding_size,
        encoder_layers=encoder_layers,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        max_decode_length=max_decode_length,
        oracle_em_epochs=oracle_em_epochs,
        oracle_factor=oracle_factor,
        sed_path=sed_path,
        tied_vocabulary=tied_vocabulary,
        batch_size=batch_size,
        beta1=beta1,
        beta2=beta2,
        dataloader_workers=dataloader_workers,
        dropout=dropout,
        eval_batch_size=eval_batch_size,
        eval_every=eval_every,
        gradient_clip=gradient_clip,
        gpu=gpu,
        label_smoothing=label_smoothing,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        optimizer=optimizer,
        save_top_k=save_top_k,
        scheduler=scheduler,
        seed=seed,
        warmup_steps=warmup_steps,
        wandb=wandb,
        dev_predictions_path=dev_predictions_path,
    )
    util.log_info(f"Best model_path: {best_model_path}")


if __name__ == "__main__":
    main()
