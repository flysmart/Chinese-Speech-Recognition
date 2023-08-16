import torch.nn as nn
import time
from .decoder import Decoder
from .encoder import Encoder


class Seq2Seq(nn.Module):
    """Sequence-to-Sequence architecture with configurable encoder and decoder.
    """

    def __init__(self, encoder=None, decoder=None):
        super(Seq2Seq, self).__init__()
        if encoder is not None and decoder is not None:
            self.encoder = encoder
            self.decoder = decoder
        else:
            self.encoder = Encoder()
            self.decoder = Decoder()

    def forward(self, padded_input, input_lengths, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        loss,att_w = self.decoder(padded_target, encoder_padded_outputs)
        return loss,att_w

    def recognize(self, input, input_length, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        # 计时
        start_time = time.perf_counter()  # 程序开始时间
        # 运行的程序
        encoder_outputs, _ = self.encoder(input.unsqueeze(0), input_length)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0],
                                                 char_list,
                                                 args)
        end_time = time.perf_counter()  # 程序结束时间
        run_time = end_time - start_time  # 程序的运行时间，单位为秒
        return nbest_hyps , run_time
