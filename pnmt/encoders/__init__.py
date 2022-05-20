"""Module defining encoders."""
from pnmt.encoders.encoder import EncoderBase
from pnmt.encoders.transformer import TransformerEncoder
from pnmt.encoders.ggnn_encoder import GGNNEncoder
from pnmt.encoders.rnn_encoder import RNNEncoder
from pnmt.encoders.cnn_encoder import CNNEncoder
from pnmt.encoders.mean_encoder import MeanEncoder
from pnmt.encoders.pre_train_encoder import PreTrainEncoder

str2enc = {"ggnn": GGNNEncoder, "rnn": RNNEncoder, "brnn": RNNEncoder,
           "cnn": CNNEncoder, "transformer": TransformerEncoder,
           "mean": MeanEncoder, "pre_train_encoder": PreTrainEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder",
           "MeanEncoder", "PreTrainEncoder", "str2enc"]
