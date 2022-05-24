from pnmt.encoders.encoder import EncoderBase
from transformers import AutoModel
import pdb


class PreTrainEncoder(EncoderBase):
    def __init__(self, encoder_type):
        super(PreTrainEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(encoder_type)
        self.pre_train = True
        self.bidirectionlal = False
    def forward(self, src, lengths=None):
        outputs = self.model(input_ids=src['input_ids'],
                             attention_mask=src['attention_mask'],
                             token_type_ids=src['token_type_ids'],
                             output_hidden_states=True)
        memory_bank = outputs['last_hidden_state']
        memory_bank = memory_bank.transpose(0, 1)
        enc_state = outputs['pooler_output']
        return enc_state, memory_bank, lengths
    @classmethod
    def from_opt(cls, opt, embeddings=None):
        return cls(opt.pre_train_encoder_type)
