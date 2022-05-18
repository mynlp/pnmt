""" Onmt NMT Model base class definition """
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder / decoder or decoder only model.
    """

    def __init__(self, encoder, decoder):
        super(BaseModel, self).__init__()

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        raise NotImplementedError

    def update_dropout(self, dropout):
        raise NotImplementedError

    def count_parameters(self, log=print):
        raise NotImplementedError


class NMTModel(BaseModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.
    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

    def pre_train_model_init_state(self, src, memory_bank, encoder_final):
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            # No direction in pre train model, we return without modification
            return hidden

        if 'LSTM' in self.decoder.rnn._get_name():  # LSTM
            self.decoder.state["hidden"] = tuple(_fix_enc_hidden(enc_hid)
                                                 for enc_hid in (encoder_final.unsqueeze(0), encoder_final.unsqueeze(0)))
            # self.decoder.state["hidden"] = tuple(_fix_enc_hidden(enc_hid)
            #                                     for enc_hid in (encoder_final.unsqueeze(0), torch.zeros(encoder_final.unsqueeze(0).shape).cuda()))
            if self.decoder.num_layers > 1:
                # reminder_layer = self.decoder.num_layers - 1
                # hidden_state = self.decoder.state["hidden"]
                # concat_zero = torch.zeros(hidden_state[0].shape).repeat((reminder_layer, 1, 1)).cuda()
                # new_hidden_state = torch.cat((hidden_state[0], concat_zero))
                # self.decoder.state["hidden"] = (new_hidden_state.cuda(),  torch.zeros(new_hidden_state.shape).cuda())
                self.decoder.state["hidden"] = (self.decoder.state["hidden"][0].repeat((self.decoder.num_layers, 1, 1)),
                                                self.decoder.state["hidden"][1].repeat((self.decoder.num_layers, 1, 1)))
        else:  # GRU
            self.decoder.state["hidden"] = (_fix_enc_hidden(encoder_final.unsqueeze(0)),)
            if self.decoder.num_layers > 1:
                self.decoder.state["hidden"] = (self.decoder.state["hidden"][0].repeat((self.decoder.num_layers, 1, 1)),)

            # Init the input feed.
        batch_size = self.decoder.state["hidden"][0].size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        self.decoder.state["input_feed"] = \
            self.decoder.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.decoder.state["coverage"] = None

    def forward(self, src, tgt, lengths, bptt=False, with_align=False, use_pre_trained_model_for_encoder=False):

        if use_pre_trained_model_for_encoder==True:
            outputs = self.encoder(input_ids=src['input_ids'],
                                   attention_mask=src['attention_mask'],
                                   token_type_ids=src['token_type_ids'],
                                   output_hidden_states=True)
            memory_bank = outputs['last_hidden_state']
            memory_bank = memory_bank.transpose(0, 1)
            enc_state = outputs['pooler_output']
        else:
            enc_state, memory_bank, lengths = self.encoder(src, lengths)

        dec_in = tgt[:-1]  # exclude last target from inputs
        if not bptt:
            if use_pre_trained_model_for_encoder:
                self.pre_train_model_init_state(src, memory_bank, enc_state)
            else:
                self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).

        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count
        """

        enc, dec = 0, 0
        for name, param in self.named_parameters():
            if 'encoder' in name:
                enc += param.nelement()
            else:
                dec += param.nelement()
        if callable(log):
            log('encoder: {}'.format(enc))
            log('decoder: {}'.format(dec))
            log('* number of parameters: {}'.format(enc + dec))
        return enc, dec


class LanguageModel(BaseModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic decoder only model.
    Currently TransformerLMDecoder is the only LM decoder implemented
    Args:
      decoder (onmt.decoders.TransformerLMDecoder): a transformer decoder
    """

    def __init__(self, encoder=None, decoder=None):
        super(LanguageModel, self).__init__(encoder, decoder)
        if encoder is not None:
            raise ValueError("LanguageModel should not be used"
                             "with an encoder")
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.
        Args:
            src (Tensor): A source sequence passed to decoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on decoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.
        Returns:
            (FloatTensor, dict[str, FloatTensor]):
            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """

        if not bptt:
            self.decoder.init_state()
        dec_out, attns = self.decoder(
            src, memory_bank=None, memory_lengths=lengths,
            with_align=with_align
        )
        return dec_out, attns

    def update_dropout(self, dropout):
        self.decoder.update_dropout(dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).
        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count
        """

        enc, dec = 0, 0
        for name, param in self.named_parameters():
            if "decoder" in name:
                dec += param.nelement()

        if callable(log):
            # No encoder in LM, seq2seq count formatting kept
            log("encoder: {}".format(enc))
            log("decoder: {}".format(dec))
            log("* number of parameters: {}".format(enc + dec))
        return enc, dec
