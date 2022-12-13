"""Training."""

import os
import time
from typing import Dict, List, Optional, Tuple

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
    model_path: str,
    *,
    train_from_path: Optional[str] = None,
    # Data format arguments.
    source_col: int = 1,
    target_col: int = 2,
    features_col: int = 3,
    source_sep: str = "",
    target_sep: str = "",
    features_sep: str = "",
    # Architecture arguments.
    arch: str = "lstm", 
    attention: bool = True,
    bidirectional: bool = True,
    dec_layers: int = 1,
    embedding_size: int = 128,
    enc_layers: int = 1,
    hidden_size: int = 512,
    max_seq_length: int = 128,
    max_dec_length: int = 128,
    nhead: int = 4,
    tied_vocabulary: bool = True,
    # Training arguments.
    batch_size: int = 128,
    beta1: float = 0.9,
    beta2: float = 0.999,
    dataloader_workers: int = 1,
    dropout: float = 0.3,
    epochs: int = 20,
    eval_batch_size: int = 128,
    eval_every: int = 5,
    gradient_clip: float = 0.0,
    gpu: bool = True,
    label_smoothing: Optional[float] = None,
    learning_rate: float = 1.0,
    lr_scheduler: Optional[str] = None,
    patience: Optional[int] = 0,
    optimizer: str = "adadelta",
    save_top_k: int = 1,
    seed: int = time.time_ns(),
    wandb: bool = False,
    warmup_steps: int = 0,
    # Training arguments specific to the transducer.
    oracle_em_epochs: int = 0,
    oracle_factor: int = 1,
    sed_path: Optional[str] = None,
) -> str:
    """Performs training, returning the path to the bset model."""
    util.seed(seed)
    # Sets up data sets, collators, and loaders.
    if target_col == 0:
        raise datasets.Error("target_col must be specified for training")
    include_features = features_col != 0
    train_set = datasets.get_dataset(
        train_data_path,
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
        include_targets=True,
    )
    train_loader = data.DataLoader(
        train_set,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
    )
    dev_set = datasets.get_dataset(
        dev_data_path,
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
    logger = [loggers.CSVLogger(output_path, name=experiment)]
    if wandb:
        logger.append(loggers.WandbLogger(project=experiment, log_model="all"))
    # ckp_callback is used later for logging the best checkpoint path.
    ckp_callback = callbacks.ModelCheckpoint(
        save_top_k=save_top_k,
        monitor="val_accuracy",
        mode="max",
        filename="model-{epoch:02d}-{val_accuracy:.2f}",
    )
    callbacks = []
    if patience is not None:
        callbacks.append(
            callbacks.early_stopping.EarlyStopping(
                monitor="val_accuracy",
                min_delta=0.00,
                patience=patience,
                verbose=False,
                mode="max",
            )
        )
    callbacks.append(ckp_callback)
    callbacks.append(callbacks.LearningRateMonitor(logging_interval="epoch"))
    callbacks.append(callbacks.TQDMProgressBar())
    trainer = pl.Trainer(
        accelerator="gpu" if gpu and torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        max_epochs=epochs,
        gradient_clip_val=gradient_clip,
        check_val_every_n_epoch=eval_every,
        enable_checkpointing=True,
        default_root_dir=output_path,
        callbacks=callbacks,
        log_every_n_steps=len(train_set) // batch_size,
        num_sanity_val_steps=0,
    )
    # So we can write indices to it before PL creates it.
    os.makedirs(trainer.loggers[0].log_dir, exist_ok=True)
    train_set.write_index(trainer.loggers[0].log_dir, experiment)
    dev_set.load_index(trainer.loggers[0].log_dir, experiment)
    # Trains model.
    model_cls = models.get_model_cls(arch, attn, include_features)
    util.log_info(f"Model: {model_cls.__name__}")
    if train_from is not None:
        util.log_info(f"Loading model from {train_from}")
        model = model_cls.load_from_checkpoint(train_from).to(device)
        util.log_info("Training...")
        trainer.fit(model, train_loader, dev_loader, ckpt_path=train_from)
    else:
        expert = None
        if arch in ["transducer"]:
            expert = models.expert.get_expert(
                train_set,
                epochs=oracle_em_epochs,
                oracle_factor=oracle_factor,
                sed_params_path=sed_params_path,
            )
        model = model_cls(
            beta1=beta1,
            beta2=beta2,
            bidirectional=bidirectional,
            dec_layers=dec_layers,
            dropout=dropout,
            enc_layers=enc_layers,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            end_idx=train_set.end_idx,
            evaluator=evaluator,
            expert=expert,
            features_idx=getattr(train_set, "features_idx", -1),
            features_vocab_size=getattr(train_set, "features_vocab_size", -1),
            label_smoothing=label_smoothing,
            lr=learning_rate,
            max_decode_len=max_decode_len,
            max_seq_len=max_seq_len,
            nhead=nhead,
            optim=optimizer,
            output_size=train_set.target_vocab_size,
            scheduler=lr_scheduler,
            pad_idx=train_set.pad_idx,
            start_idx=train_set.start_idx,
            vocab_size=train_set.source_vocab_size,
            warmup_steps=warmup_steps,
        ).to(device)
        util.log_info("Training...")
        trainer.fit(model, train_loader, dev_loader)
    util.log_info("Training complete")
    return ckp_callback.best_model_path


@click.command()
@click.option("--train-data-path", required=True)
@click.option("--dev-data-path", required=True)
@click.option("--dev-predictions-path")
@click.option("--source-col", type=int, default=1)
@click.option("--target-col", type=int, default=2)
@click.option(
    "--features-col",
    type=int,
    default=3,
    help="0 indicates no feature column should be used",
)
@click.option("--source-sep", type=str, default="")
@click.option("--target-sep", type=str, default="")
@click.option("--features-sep", type=str, default=";")
@click.option("--tied-vocabulary/--no-tied-vocabulary", default=True)
@click.option("--output-path", required=True)
@click.option("--dataloader-workers", type=int, default=1)
@click.option("--experiment-name", required=True)
@click.option("--seed", type=int, default=time.time_ns())
@click.option("--epochs", type=int, default=20)
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
    required=True,
)
@click.option(
    "--oracle-em-epochs",
    type=int,
    default=0,
    help="Number of EM epochs (`--arch transducer` only)",
)
@click.option(
    "--oracle-factor",
    type=int,
    default=1,
    help="Roll-in schedule parameter (`--arch transducer` only)",
)
@click.option(
    "--sed-params-path",
    type=str,
    default=None,
    help="Path to SED parameters (`transducer` only)",
)
@click.option("--patience", type=int)
@click.option("--learning-rate", type=float, required=True)
@click.option("--label-smoothing", type=float)
@click.option("--gradient-clip", type=float, default=0.0)
@click.option("--batch-size", type=int, default=16)
@click.option("--eval-batch-size", type=int, default=1)
@click.option("--embedding-size", type=int, default=128)
@click.option("--hidden-size", type=int, default=256)
@click.option("--dropout", type=float, default=0.3)
@click.option("--enc-layers", type=int, default=1)
@click.option("--dec-layers", type=int, default=1)
@click.option("--max-seq-len", type=int, default=128)
@click.option("--nhead", type=int, default=4)
@click.option("--dropout", type=float, default=0.1)
@click.option("--optimizer", default="adadelta")
@click.option(
    "--beta1",
    default=0.9,
    type=float,
    help="beta1 (`--optimizer adam` only)",
)
@click.option(
    "--beta2",
    default="0.999",
    type=float,
    help="beta2 (`--optimizer adam` only)",
)
@click.option("--warmup-steps", default=1)
@click.option("--lr-scheduler")
@click.option(
    "--train-from",
    help="Path to checkpoint to continue training from",
)
@click.option("--bidirectional/--no-bidirectional", type=bool, default=True)
@click.option(
    "--attn/--no-attn",
    type=bool,
    default=True,
    help="Use attention (`--arch lstm` only)",
)
@click.option("--max-decode-len", type=int, default=128)
@click.option("--save-top-k", type=int, default=1)
@click.option("--eval-every", type=int, default=5)
@click.option("--gpu/--no-gpu", default=True)
@click.option("--wandb/--no-wandb", default=False)
def main(
    train_data_path,
    dev_data_path,
    dev_predictions_path,
    tied_vocabulary,
    source_col,
    target_col,
    features_col,
    source_sep,
    target_sep,
    features_sep,
    output_path,
    dataloader_workers,
    experiment_name,
    seed,
    epochs,
    arch,
    oracle_em_epochs,
    oracle_factor,
    sed_params_path,
    patience,
    learning_rate,
    label_smoothing,
    gradient_clip,
    batch_size,
    eval_batch_size,
    embedding_size,
    hidden_size,
    dropout,
    enc_layers,
    dec_layers,
    max_seq_len,
    nhead,
    optimizer,
    beta1,
    beta2,
    warmup_steps,
    lr_scheduler,
    train_from,
    bidirectional,
    attn,
    max_decode_len,
    save_top_k,
    eval_every,
    gpu,
    wandb,
):
    """Training.

    Args:
        train_data_path (_type_): _description_
        dev_data_path (_type_): _description_
        dev_predictions_path (_type_): _description_
        source_col (_type_): _description_
        target_col (_type_): _description_
        features_col (_type_): _description_
        source_sep (_type_): _description_
        target_sep (_type_): _description_
        features_sep (_type_): _description_
        tied_vocabulary (_type_): _description_
        output_path (_type_): _description_
        dataset (_type_): _description_
        dataloader_workers (_type_): _description_
        experiment_name (_type_): _description_
        seed (_type_): _description_
        epochs (_type_): _description_
        arch (_type_): _description_
        oracle_em_epochs (_type_): _description_
        oracle_factor (_type_): _description_
        sed_params_path (_type_): _description_
        patience (_type_): _description_
        learning_rate (_type_): _description_
        label_smoothing (_type_): _description_
        gradient_clip (_type_): _description_
        batch_size (_type_): _description_
        eval_batch_size (_type_): _description_
        embedding_size (_type_): _description_
        hidden_size (_type_): _description_
        dropout (_type_): _description_
        enc_layers (_type_): _description_
        dec_layers (_type_): _description_
        max_seq_len: (_type_) _description_
        nhead (_type_): _description_
        optimizer (_type_): _description_
        beta1 (_type_): _description_
        beta2 (_type_): _description_
        warmup_steps (_type_): _description_
        scheduler (_type_): _description_
        train_from (_type_): _description_
        bidirectional (_type_): _description_
        attn (_type_): _description_
        max_decode_len (_type_): _description_
        save_top_k (_type_): _description_
        eval_every (_type_): _description_
        gpu (_type_): _description_
        wandb (_type_): _description_
    """
    util.log_info("Arguments:")
    for arg, val in click.get_current_context().params.items():
        util.log_info(f"\t{arg}: {val!r}")
    best_model_path = train(...)
    util.log_info(f"Best model_path: {best_model_path}")
    # If specified, write training and/or dev set predictions.
    # TODO: Add beam-width option so we can make predictions with beam search.
    if train_predictions_path or dev_predictions_path:
        model_cls = models.get_model_cls(arch, attn, include_features)
        model = model_cls.load_from_checkpoint(ckp_callback.best_model_path).to(device)
    if train_predictions_path:
        predict.predict(
            model,
            train_loader,
            train_predictions_path,
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
    if dev_predictions_path:
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


if __name__ == "__main__":
    main()
