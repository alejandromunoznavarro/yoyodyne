"""Transformer model classes."""

import math
from typing import Optional

import torch
from torch import nn

from .. import evaluators
from . import base, positional_encoding


class Error(Exception):
    pass


class TransformerEncoderDecoder(base.BaseEncoderDecoder):
    """Transformer encoder-decoder."""

    # Training arguments
    attention_heads: int
    beta1: float
    beta2: float
    decoder_layers: int
    embedding_size: int
    encoder_layers: int
    end_idx: int
    evaluator: evaluators.Evaluator
    hidden_size: int
    label_smoothing: Optional[float]
    learning_rate: float
    max_decode_length: int
    max_sequence_length: int
    optimizer: str
    output_size: int
    pad_idx: int
    scheduler: Optional[str]
    start_idx: int
    vocab_size: int
    warmup_steps: int
    # Decoding arguments.
    beam_width: Optional[int]
    # Components of the module.
    dropout: nn.Dropout
    source_embeddings: nn.Embedding
    target_embeddings: nn.Embedding
    positional_encoding: positional_encoding.PositionalEncoding
    log_softmax: nn.LogSoftmax
    encoder: nn.TransformerEncoder
    decoder: nn.TransformerDecoder
    classifier: nn.Linear

    def __init__(
        self,
        *,
        attention_heads,
        beta1,
        beta2,
        decoder_layers,
        dropout,
        embedding_size,
        encoder_layers,
        end_idx,
        evaluator,
        hidden_size,
        label_smoothing,
        learning_rate,
        max_decode_length,
        max_sequence_length,
        optimizer,
        output_size,
        pad_idx,
        scheduler,
        start_idx,
        vocab_size,
        warmup_steps,
        beam_width=None,
        **kwargs,
    ):
        """Initializes the encoder-decoder with attention.

        Args:
            **kwargs: ignored.
        """
        self.attention_heads = attention_heads
        self.beta1 = beta1
        self.beta2 = beta2
        self.decoder_layers = decoder_layers
        self.embedding_size = embedding_size
        self.encoder_layers = encoder_layers
        self.end_idx = end_idx
        self.evaluator = evaluator
        self.hidden_size = hidden_size
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate
        self.max_decode_length = max_decode_length
        self.max_sequence_length = max_sequence_length
        self.optimizer = optimizer
        self.output_size = output_size
        self.pad_idx = pad_idx
        self.scheduler = scheduler
        self.start_idx = start_idx
        self.vocab_size = vocab_size
        self.warmup_steps = warmup_steps
        self.beam_width = beam_width
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.source_embeddings = self.init_embeddings(
            vocab_size, self.embedding_size, self.pad_idx
        )
        self.target_embeddings = self.init_embeddings(
            output_size, self.embedding_size, self.pad_idx
        )
        self.positional_encoding = positional_encoding.PositionalEncoding(
            self.embedding_size, self.pad_idx, self.max_sequence_length
        )
        self.log_softmax = nn.LogSoftmax(dim=2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            dim_feedforward=self.hidden_size,
            nhead=self.attention_heads,
            dropout=dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.encoder_layers,
            norm=nn.LayerNorm(self.embedding_size),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embedding_size,
            dim_feedforward=self.hidden_size,
            nhead=self.attention_heads,
            dropout=dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.decoder_layers,
            norm=nn.LayerNorm(self.embedding_size),
        )
        self.classifier = nn.Linear(self.embedding_size, output_size)
        self.loss_func = self.get_loss_func("mean")
        # Saves hyperparameters for PL checkpointing.
        self.save_hyperparameters()

    def init_embeddings(
        self, num_embeddings: int, embedding_size: int, pad_idx: int
    ) -> nn.Embedding:
        """Initializes the embedding layer.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.
            pad_idx (int): index of pad symbol.

        Returns:
            nn.Embedding: embedding layer.
        """
        return self._xavier_embedding_initialization(
            num_embeddings, embedding_size, pad_idx
        )

    def source_embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the source symbols and adds positional encodings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x seq_len.

        Returns:
            embedded (torch.Tensor): embedded tensor of shape
                B x seq_len x embed_dim.
        """
        word_embedding = self.source_embeddings(symbols) * math.sqrt(
            self.embedding_size
        )
        positional_embedding = self.positional_encoding(symbols)
        out = self.dropout(word_embedding + positional_embedding)
        return out

    def target_embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the target symbols and adds positional encodings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x seq_len.

        Returns:
            embedded (torch.Tensor): embedded tensor of shape
                B x seq_len x embed_dim.
        """
        word_embedding = self.target_embeddings(symbols) * math.sqrt(
            self.embedding_size
        )
        positional_embedding = self.positional_encoding(symbols)
        out = self.dropout(word_embedding + positional_embedding)
        return out

    def encode(
        self, source: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encodes the source with the TransformerEncoder.

        Args:
            source (torch.Tensor).
            source_mask (torch.Tensor).

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        embedding = self.source_embed(source)
        return self.encoder(embedding, src_key_padding_mask=source_mask)

    def decode(
        self,
        enc_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes the logits for each step of the output sequence.

        Args:
            enc_hidden (torch.Tensor): source encoder hidden state of shape
                                   B x seq_len x hidden_size
            source_mask (torch.Tensor): encoder hidden state mask.
            target (torch.Tensor): current state of targets, which may be the
                full target, or previous decoded, of shape
                B x seq_len x hidden_size.
            target_mask (torch.Tensor): target mask.

        Returns:
            _type_: log softmax over targets.
        """
        target_embedding = self.target_embed(target)
        target_seq_len = target_embedding.size(1)
        # -> seq_len x seq_len.
        causal_mask = self.generate_square_subsequent_mask(target_seq_len).to(
            self.device
        )
        # -> B x seq_len x embedding_size
        dec_hidden = self.decoder(
            target_embedding,
            enc_hidden,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_mask,
            memory_key_padding_mask=source_mask,
        )
        # -> B x seq_len x vocab_size.
        output = self.classifier(dec_hidden)
        output = self.log_softmax(output)
        return output

    def _decode_greedy(
        self, enc_hidden: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        # The output distributions to be returned.
        outputs = []
        batch_size = enc_hidden.size(0)
        # The predicted symbols at each iteration.
        preds = [
            torch.tensor(
                [self.start_idx for _ in range(enc_hidden.size(0))],
                device=self.device,
            )
        ]
        # Tracking when each sequence has decoded an EOS.
        finished = torch.zeros(batch_size, device=self.device)
        for _ in range(self.max_decode_length):
            target_tensor = torch.stack(preds, dim=1)
            # Uses a dummy mask of all ones.
            target_mask = torch.ones_like(
                target_tensor, dtype=torch.float, device=self.device
            )
            target_mask = target_mask == 0
            output = self.decode(
                enc_hidden, source_mask, target_tensor, target_mask
            )
            # We only care about the last prediction in the sequence.
            last_output = output[:, -1, :]
            outputs.append(last_output)
            pred = self._get_predicted(last_output.unsqueeze(1))
            preds.append(pred.squeeze(1))
            # Updates to track which sequences have decoded an EOS.
            finished = torch.logical_or(finished, preds[-1] == self.end_idx)
            # Break when all batches predicted an EOS symbol.
            if finished.all():
                break
        # -> B x vocab_size x seq_len
        return torch.stack(outputs).transpose(0, 1).transpose(1, 2)

    def forward(self, batch: base.Batch) -> torch.Tensor:
        """Runs the encoder-decoder.

        Args:
            batch (Tuple[torch.Tensor, ...]): Tuple of tensors in the batch
                of shape (source, source_mask, target, target_mask) during
                training or shape (source, source_mask) during inference.

        Returns:
            torch.Tensor.
        """
        # Training mode with targets.
        if len(batch) == 4:
            source, source_mask, target, target_mask = batch
            # Initializes the start symbol for decoding.
            starts = (
                torch.LongTensor([self.start_idx])
                .to(self.device)
                .repeat(target.size(0))
                .unsqueeze(1)
            )
            target = torch.cat((starts, target), dim=1)
            target_mask = torch.cat(
                (starts == self.pad_idx, target_mask), dim=1
            )
            enc_hidden = self.encode(source, source_mask)
            output = self.decode(enc_hidden, source_mask, target, target_mask)
            # -> B x vocab_size x seq_len
            output = output.transpose(1, 2)[:, :, :-1]
        # No targets given at inference.
        elif len(batch) == 2:
            source, source_mask = batch
            enc_hidden = self.encode(source, source_mask)
            # -> B x vocab_size x seq_len.
            output = self._decode_greedy(enc_hidden, source_mask)
        else:
            raise Error(f"Batch of {len(batch)} elements is invalid")
        return output

    def generate_square_subsequent_mask(self, length: int) -> torch.Tensor:
        """Generates the target mask so the model cannot see future states.

        Args:
            length (int): length of the sequence.

        Returns:
            torch.Tensor: mask of shape length x length.
        """
        return torch.triu(torch.full((length, length), -math.inf), diagonal=1)


class FeatureInvariantTransformerEncoderDecoder(TransformerEncoderDecoder):
    """Transformer encoder-decoder with feature invariance.

    After:
        Wu, S., Cotterell, R., and Hulden, M. 2021. Applying the transformer to
        character-level transductions. In Proceedings of the 16th Conference of
        the European Chapter of the Association for Computational Linguistics:
        Main Volume, pages 1901-1907.
    """

    features_idx: int

    def __init__(self, *args, features_idx, **kwargs):
        super().__init__(*args, **kwargs)
        # Distinguishes source vs. feature symbols.
        self.features_idx = features_idx
        self.type_embedding = self.init_embeddings(
            2, self.embedding_size, self.pad_idx
        )

    def source_embed(self, symbols: torch.Tensor) -> torch.Tensor:
        """Embeds the source symbols.

        This adds positional encodings and special embeddings.

        Args:
            symbols (torch.Tensor): batch of symbols to embed of shape
                B x seq_len.

        Returns:
            embedded (torch.Tensor): embedded tensor of shape
                B x seq_len x embed_dim.
        """
        # Distinguishes features and chars.
        char_mask = (symbols < self.features_idx).long()
        # 1 or 0.
        type_embedding = math.sqrt(self.embedding_size) * self.type_embedding(
            char_mask
        )
        word_embedding = self.source_embeddings(symbols) * math.sqrt(
            self.embedding_size
        )
        positional_embedding = self.positional_encoding(
            symbols, mask=char_mask
        )
        out = self.dropout(
            word_embedding + positional_embedding + type_embedding
        )
        return out
