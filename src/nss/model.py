import math
import torch
import torch.nn as nn
from tools import config
from tools import state_letters, theorem_letters

"""Model definition."""


class Embedding(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model, padding_idx=0)  # default padding 0

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, seq_len])
        :return result: torch.Size([batch_size, seq_len, d_model])
        """
        return self.emb(x) * math.sqrt(self.d_model)  # make the variance of `emb` distribution becomes 1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=config["data"]["max_len"]):
        """Standard positional encoding from original transformer."""

        super(PositionalEncoding, self).__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor([10000.0])) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 'pe' will be retained when model saving and loading, but it will not be updated during the training.
        self.register_buffer('pe', pe)  # torch.Size([max_len, d_model])

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, seq_len, d_model])
        :return result: torch.Size([batch_size, seq_len, d_model])
        """
        pe = self.pe[:x.size(1), :].unsqueeze(0)  # [1, seq_len, d_model]
        x = x + pe.masked_fill(x == 0, 0)  # [batch_size, seq_len, d_model]
        return x


class GatedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, no_gate):
        """d_q = d_k = d_v = d_head = d_model / h"""
        super(GatedMultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.d_head = d_model // h
        self.h = h
        self.no_gate = no_gate

        self.q_proj = nn.Linear(self.d_model, self.d_model * 2, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, seq_len, d_model])
        :return result: torch.Size([batch_size, seq_len, d_model])
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # QKV Linear Projections
        query, gate_score = torch.split(self.q_proj(x), self.d_model, dim=-1)  # [batch_size, seq_len, d_model]
        query = query.view(batch_size, seq_len, self.h, self.d_head).transpose(1, 2)  # [batch_size, h, seq_len, d_k]
        key = self.k_proj(x).view(batch_size, seq_len, self.h, self.d_head).transpose(1, 2)
        value = self.v_proj(x).view(batch_size, seq_len, self.h, self.d_head).transpose(1, 2)

        # Scaled Product Dot-Product Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.softmax(dim=-1)  # [batch_size, h, seq_len, seq_len]
        x = torch.matmul(scores, value)

        # Multi-Head Concatenation
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # [batch_size, seq_len, d_model]

        # Gating
        if not self.no_gate:
            x = x * torch.sigmoid(gate_score)  # [batch_size, seq_len, d_model]

        # Linear Projections
        x = self.o_proj(x)  # [batch_size, seq_len, d_model]

        return x


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, seq_len, d_model])
        :return result: torch.Size([batch_size, seq_len, d_model])
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return x


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.ffd = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, seq_len, d_model])
        :return result: torch.Size([batch_size, seq_len, d_model])
        """
        return self.ffd(x)


class BasicModule(nn.Module):
    def __init__(self, d_model, h, p_drop, no_gate):
        """
        Sentence encoder, encode sentence with n words to 1 dimension-fixed vector.
        :param d_model: Embedding dim.
        :param h: Head number in MultiHeadAttention.
        :param p_drop: Dropout rate.
        """
        super(BasicModule, self).__init__()
        self.dropout = nn.Dropout(p_drop)
        self.attention = GatedMultiHeadAttention(d_model, h, no_gate)
        self.layer_norm_attn = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_model * 4)
        self.layer_norm_ffd = LayerNorm(d_model)

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, seq_len, d_model])
        :return x: torch.Size([batch_size, seq_len, d_model])
        """
        x = self.layer_norm_attn(x + self.dropout(self.attention(x)))  # gated multi-head attention and layer norm
        x = self.layer_norm_ffd(x + self.dropout(self.feed_forward(x)))  # feed_forward and layer norm
        return x


class TheoremPredictor(nn.Module):
    def __init__(self, d_image, vocab_state, vocab_theorem, d_model, M, N, h, p_drop, text_only, forward_only, no_gate):
        """
        Sentence encoder, encode sentence with n words to 1 dimension-fixed vector.
        :param d_model: Embedding dim.
        :param h: Head number in MultiHeadAttention.
        :param N: Number of MultiHeadAttention.
        :param p_drop: Dropout rate.
        """
        super(TheoremPredictor, self).__init__()
        self.text_only = text_only
        self.forward_only = forward_only
        self.M = M
        self.N = N

        self.image_embedding = nn.Linear(d_image, d_model, bias=False)
        self.state_embedding = Embedding(vocab_state, d_model)
        self.pe = PositionalEncoding(d_model)

        self.encoder = nn.ModuleList([BasicModule(d_model, h, p_drop, no_gate) for _ in range(M)])

        self.forward_tp = nn.ModuleList([BasicModule(d_model, h, p_drop, no_gate) for _ in range(N)])
        self.linear_forward_tp = nn.Linear(d_model, vocab_theorem, bias=False)

        self.backward_tp = nn.ModuleList([BasicModule(d_model, h, p_drop, no_gate) for _ in range(N)])
        self.linear_backward_tp = nn.Linear(d_model, vocab_theorem, bias=False)

    def forward(self, state, image):
        """
        :param image: torch.Size([batch_size, image_seq_len, d_image])
        :param state: torch.Size([batch_size, state_seq_len])
        :return result: torch.Size([batch_size, d_model])
        """
        problem_encoding = self.state_embedding(state)  # [batch_size, state_seq_len, d_model]
        if not self.text_only:
            image_embedding = self.image_embedding(image)  # [batch_size, image_seq_len, d_model]
            problem_encoding = torch.cat([problem_encoding, image_embedding], dim=1)
        problem_encoding = self.pe(problem_encoding)  # positional encoding

        for i in range(self.M):  # problem encoding
            problem_encoding = self.encoder[i](problem_encoding)

        forward_encoding = problem_encoding  # forward theorem predictor
        for i in range(self.N):
            forward_encoding = self.forward_tp[i](forward_encoding)
        # print('forward_encoding.size()', forward_encoding.size())

        # forward_theorem_logits = self.linear_forward_tp(forward_encoding[:, 0, :])  # use first token
        forward_theorem_logits = self.linear_forward_tp(forward_encoding.mean(dim=1))  # use first token

        # print('forward_theorem_logits.size()', forward_theorem_logits.size())
        # print()

        if self.forward_only:
            return forward_theorem_logits

        backward_encoding = problem_encoding  # backward theorem predictor
        for i in range(self.N):
            backward_encoding = self.backward_tp[i](backward_encoding)
        # print('backward_encoding.size()', backward_encoding.size())

        # backward_theorem_logits = self.linear_backward_tp(backward_encoding[:, 0, :])  # use first token
        backward_theorem_logits = self.linear_backward_tp(backward_encoding.mean(dim=1))  # use first token

        # print('backward_theorem_logits.size()', backward_theorem_logits.size())
        # print()

        return forward_theorem_logits, backward_theorem_logits


def make_model(text_only=False, forward_only=False, no_gate=False, small_model=False):
    model = TheoremPredictor(
        d_image=config["model"]["d_image"],
        vocab_state=len(state_letters),
        vocab_theorem=len(theorem_letters),
        d_model=config["model"]["d_model"] if not small_model else int(config["model"]["d_model"] / 2),
        M=config["model"]["M"],
        N=config["model"]["N"],
        h=config["model"]["h"],
        p_drop=config["model"]["p_drop"],
        text_only=text_only,
        forward_only=forward_only,
        no_gate=no_gate
    )
    return model


def show_parameters():
    """
    Params (standard / small): 8,846,080 (8.85 M) / 2293120 (2.29 M)
    Memory (standard / small): 33.75 MB / 8.75 MB
    """
    m = make_model(small_model=True)
    total_params = sum(p.numel() for p in m.parameters())  # 参数总数
    param_memory = sum(p.numel() * p.element_size() for p in m.parameters())  # 占用字节数
    print("Params: {} ({:.2f} M), Memory: {:.2f} MB.".format(
        total_params, total_params / 1000000, param_memory / 1024 / 1024)
    )


if __name__ == '__main__':
    show_parameters()
