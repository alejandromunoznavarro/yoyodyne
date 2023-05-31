"""LSTM model classes."""

import argparse
import heapq
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .. import batches, defaults
from . import base, EncoderMismatchError
from .encoders import LSTMEncoder, LSTMDecoder, LSTMAttentiveDecoder


class LSTMEncoderDecoder(base.BaseEncoderDecoder):
    """LSTM encoder-decoder without attention.

    We achieve this by concatenating the last (non-padding) hidden state of
    the encoder to the decoder hidden state."""
    # Model arguments.
    bidirectional: bool
    # Constructed inside __init__.
    h0: nn.Parameter
    c0: nn.Parameter
    decoder: nn.LSTM
    classifier: nn.Linear
    log_softmax: nn.LogSoftmax

    def __init__(
        self,
        *args,
        bidirectional,
        **kwargs,
    ):
        """Initializes the encoder-decoder without attention.

        Args:
            *args: passed to superclass.
            bidirectional (bool).
            **kwargs: passed to superclass.
        """
        super().__init__(*args, bidirectional, **kwargs)
        self.bidirectional = bidirectional
        # Initial hidden state whose parameters are shared across all examples.
        self.h0 = nn.Parameter(torch.rand(self.hidden_size))
        self.c0 = nn.Parameter(torch.rand(self.hidden_size))

    def get_decoder(self):
        return LSTMDecoder(
                pad_idx=self.pad_idx,
                start_idx=self.start_idx,
                end_idx=self.end_idx,
                decoder_size=self.source_encoder.output_size + self.embedding_size,
                num_embeddings=self.output_size,
                dropout=self.dropout,
                bidirectional=False,
                embedding_size=self.embedding_size,
                layers=self.decoder_layers,
                hidden_size=self.hidden_size
        )

    def init_hiddens(
        self, batch_size: int, num_layers: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes the hidden state to pass to the LSTM.

        Note that we learn the initial state h0 as a parameter of the model.

        Args:
            batch_size (int).
            num_layers (int).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: hidden cells for LSTM
                initialization.
        """
        return (
            self.h0.repeat(num_layers, batch_size, 1),
            self.c0.repeat(num_layers, batch_size, 1),
        )

    def decode(
        self,
        batch_size: int,
        encoder_mask: torch.Tensor,
        encoder_out: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence given the encoded input.

        This initializes an <EOS> tensor, and decodes using teacher forcing
        when training, or else greedily.

        Args:
            batch_size (int).
            encoder_mask (torch.Tensor): mask for the batch of encoded inputs.
            encoder_out (torch.Tensor): batch of encoded inputs.
            target (torch.Tensor, optional): target symbols; if None, then we
                decode greedily with 'student forcing'.

        Returns:
            predictions (torch.Tensor): tensor of predictions of shape
                seq_len x batch_size x output_size.
        """
        # Initializes hidden states for decoder LSTM.
        decoder_hiddens = self.init_hiddens(batch_size, self.decoder_layers)
        # Feed in the first decoder input, as a start tag.
        # -> B x 1.
        decoder_input = (
            torch.tensor(
                [self.start_idx], device=self.device, dtype=torch.long
            )
            .repeat(batch_size)
            .unsqueeze(1)
        )
        predictions = []
        num_steps = (
            target.size(1) if target is not None else self.max_target_length
        )
        # Tracks when each sequence has decoded an EOS.
        finished = torch.zeros(batch_size, device=self.device)
        for t in range(num_steps):
            # pred: B x 1 x output_size.
            output, decoder_hiddens = self.decoder(
                decoder_input, decoder_hiddens, encoder_out, encoder_mask
            )
            predictions.append(output.squeeze(1))
            # If we have a target (training) then the next input is the gold
            # symbol for this step (i.e., teacher forcing).
            if target is not None:
                decoder_input = target[:, t].unsqueeze(1)
            # Otherwise, it must be inference time and we pass the top pred to
            # the next next timestep (i.e., student forcing, greedy decoding).
            else:
                decoder_input = self._get_predicted(output)
                # Updates to track which sequences have decoded an EOS.
                finished = torch.logical_or(
                    finished, (decoder_input == self.end_idx)
                )
                # Breaks when all batches predicted an EOS symbol.
                if finished.all():
                    break
        predictions = torch.stack(predictions)
        return predictions

    def beam_decode(
        self,
        batch_size: int,
        encoder_mask: torch.Tensor,
        encoder_out: torch.Tensor,
        beam_width: int,
        n: int = 1,
        return_confidences: bool = False,
    ) -> Union[Tuple[List, List], List]:
        """Beam search with beam_width.

        Note that we assume batch size is 1.

        Args:
            batch_size (int).
            encoder_mask (torch.Tensor).
            encoder_out (torch.Tensor): encoded inputs.
            beam_width (int): size of the beam.
            n (int): number of hypotheses to return.
            return_confidences (bool, optional): additionally return the
                likelihood of each hypothesis.

        Returns:
            Union[Tuple[List, List], List]: _description_
        """
        # TODO: only implemented for batch size of 1. Implement batch mode.
        if batch_size != 1:
            raise NotImplementedError(
                "Beam search is not implemented for batch_size > 1"
            )
        # Initializes hidden states for decoder LSTM.
        decoder_hiddens = self.init_hiddens(
            encoder_out.size(0), self.decoder_layers
        )
        # log likelihood, last decoded idx, all likelihoods,  hiddens tensor.
        histories = [[0.0, [self.start_idx], [0.0], decoder_hiddens]]
        for t in range(self.max_target_length):
            # List that stores the heap of the top beam_width elements from all
            # beam_width x output_size possibilities
            likelihoods = []
            hypotheses = []
            # First accumulates all beam_width softmaxes.
            for (
                beam_likelihood,
                beam_idxs,
                char_likelihoods,
                decoder_hiddens,
            ) in histories:
                # Does not keep decoding a path that has hit EOS.
                if len(beam_idxs) > 1 and beam_idxs[-1] == self.end_idx:
                    fields = [
                        beam_likelihood,
                        beam_idxs,
                        char_likelihoods,
                        decoder_hiddens,
                    ]
                    # TODO: Beam search with beam_width.
                    # TODO: Replace heapq with torch.max or similar?
                    heapq.heappush(hypotheses, fields)
                    continue
                # Feeds in the first decoder input, as a start tag.
                # -> batch_size x 1
                decoder_input = torch.tensor(
                    [beam_idxs[-1]], device=self.device, dtype=torch.long
                ).unsqueeze(1)
                output, decoder_hiddens = self.decoder(
                    decoder_input, decoder_hiddens, encoder_out, encoder_mask
                )
                likelihoods.append(
                    (
                        predictions,
                        beam_likelihood,
                        beam_idxs,
                        char_likelihoods,
                        decoder_hiddens,
                    )
                )
            # Constrains the next step to beamsize.
            for (
                predictions,
                beam_likelihood,
                beam_idxs,
                char_likelihoods,
                decoder_hiddens,
            ) in likelihoods:
                # This is 1 x 1 x output_size since we fixed batch size to 1.
                # We squeeze off the fist 2 dimensions to get a tensor of
                # output_size.
                predictions = predictions.squeeze(0).squeeze(0)
                for j, prob in enumerate(predictions):
                    if return_confidences:
                        cl = char_likelihoods + [prob]
                    else:
                        cl = char_likelihoods
                    if len(hypotheses) < beam_width:
                        fields = [
                            beam_likelihood + prob,
                            beam_idxs + [j],
                            cl,
                            decoder_hiddens,
                        ]
                        heapq.heappush(hypotheses, fields)
                    else:
                        fields = [
                            beam_likelihood + prob,
                            beam_idxs + [j],
                            cl,
                            decoder_hiddens,
                        ]
                        heapq.heappushpop(hypotheses, fields)
            # Takes the top beam hypotheses from the heap.
            histories = heapq.nlargest(beam_width, hypotheses)
            # If the top n hypotheses are full sequences, break.
            if all([h[1][-1] == self.end_idx for h in histories]):
                break
        # Returns the top-n hypotheses.
        histories = heapq.nlargest(n, hypotheses)
        predictions = torch.tensor([h[1] for h in histories], self.device)
        # Converts shape to that of `decode`: seq_len x B x output_size.
        predictions = predictions.unsqueeze(0).transpose(0, 2)
        if return_confidences:
            return (predictions, torch.tensor([h[2] for h in histories]))
        else:
            return predictions

    def forward(self, batch: batches.PaddedBatch) -> torch.Tensor:
        """Runs the encoder-decoder model.

        Args:
            batch (batches.PaddedBatch).

        Returns:
            predictions (torch.Tensor): tensor of predictions of shape
                (seq_len, batch_size, output_size).
        """
        encoder_out, (H, C) = self.source_encoder(batch.source)
        if self.beam_width is not None and self.beam_width > 1:
            predictions = self.beam_decode(
                len(batch),
                batch.source.mask,
                encoder_out,
                beam_width=self.beam_width,
            )
        else:
            predictions = self.decode(
                len(batch), batch.source.mask, encoder_out, batch.target.padded
            )
        # -> B x seq_len x output_size.
        predictions = predictions.transpose(0, 1)
        return predictions

    def check_encoder_compatibility(source_encoder_cls, feature_encoder_cls=None):
        if feature_encoder_cls is not None:
            raise EncoderMismatchError("This model does not support a separate feature encoder.")
        if source_encoder_cls not in [LSTMEncoder]:
            raise EncoderMismatchError("This model does not support provided encoder type.")

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        """Adds LSTM configuration options to the argument parser.

        Args:
            parser (argparse.ArgumentParser).
        """
        parser.add_argument(
            "--bidirectional",
            action="store_true",
            default=defaults.BIDIRECTIONAL,
            help="Uses a bidirectional encoder "
            "(LSTM-backed architectures only. Default: %(default)s.",
        )
        parser.add_argument(
            "--no_bidirectional",
            action="store_false",
            dest="bidirectional",
        )


class AttentiveLSTMEncoderDecoder(LSTMEncoderDecoder):
    """LSTM encoder-decoder with attention."""
    def get_decoder(self):
        return LSTMAttentiveDecoder(
                pad_idx=self.pad_idx,
                start_idx=self.start_idx,
                end_idx=self.end_idx,
                decoder_size=self.source_encoder.output_size,
                num_embeddings=self.output_size,
                dropout=self.dropout,
                bidirectional=False,
                embedding_size=self.embedding_size,
                layers=self.decoder_layers,
                hidden_size=self.hidden_size
        )