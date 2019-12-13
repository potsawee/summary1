import sys
import torch
import torch.nn as nn
import numpy as np
import pdb

class PointerGeneratorNetwork(nn.Module):
    def __init__(self, args, device):
        super(PointerGeneratorNetwork, self).__init__()
        self.device = device
        self.num_words_meeting = args['num_words_meeting']
        self.summary_length = args['summary_length']
        self.vocab_size = args['vocab_size']
        self.embedding_dim = args['embedding_dim']
        self.hidden_size = args['rnn_hidden_size']
        self.num_layers_enc = args['num_layers_enc']
        self.num_layers_dec = args['num_layers_dec']
        self.dropout = args['dropout']

        # Encoder
        self.encoder_embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.encoder_rnn = nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layers_enc, bias=True,
                                   batch_first=True, dropout=self.dropout, bidirectional=True)
        # Decoder
        self.decoder_embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.decoder_rnn = nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layers_dec, bias=True,
                                   batch_first=True, dropout=self.dropout, bidirectional=False)
        # Attention Mechanism
        self.attention_h = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)
        self.attention_s = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.attention_v = nn.Linear(self.hidden_size, 1, bias=False)
        self.attention_softmax = nn.Softmax(dim=-1)

        # Linear Output
        self.decoder_linear1 = nn.Linear(3*self.hidden_size, self.hidden_size, bias=True)
        self.decoder_linear2 = nn.Linear(self.hidden_size, self.vocab_size, bias=True)
        self.decoder_softmax = nn.Softmax(dim=-1)

        # Pointer - Generator Mechanism
        self.pointer_c = nn.Linear(2*self.hidden_size, 1, bias=True)
        self.pointer_s = nn.Linear(self.hidden_size, 1, bias=False)
        self.pointer_x = nn.Linear(self.embedding_dim, 1, bias=False)

        # LogSoftmax
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.param_init()
        self.to(device)

    def param_init(self):
        # Initialisation
        # zero out the bias term
        # don't zero out LayerNorm term e.g. transformer_encoder.layers.0.norm1.weight
        for name, p in self.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else:
                # if name[-4:] == 'bias': p.data.zero_()
                if 'bias' in name: nn.init.zeros_(p)

    def forward(self, input, u_len, w_len, target):
        # Encoder
        # PGN - doesn't have hierarchical structure so
        #       reshape the input to [batch_size, num_words_meeting]
        batch_size = input.size(0)
        input_pgn = torch.zeros((batch_size, self.num_words_meeting),dtype=torch.long).to(self.device)
        uw_len = torch.zeros((batch_size), dtype=torch.long).to(self.device)
        # no [SEP] added between utterances! --- just concat all the utterances in a meeting dialogue
        for i in range(batch_size):
            wi = 0
            for j in range(u_len[i]):
                wl = w_len[i,j]
                if wi+wl > self.num_words_meeting: break
                input_pgn[i,wi:wi+wl] = input[i,j,:wl]
                wi += wl
            uw_len[i] = wi

        # Encoder Forward
        enc_embed = self.encoder_embedding(input_pgn)
        enc_output, _ = self.encoder_rnn(enc_embed) # enc_output => [batch_size, num_words_meeting, hidden_size*2]

        # Decoder Forward
        dec_embed = self.decoder_embedding(target)
        dec_output, _ = self.decoder_rnn(dec_embed)

        # Attention Mechanism --- to get context vector
        Wh_ht = self.attention_h(enc_output)
        Ws_st = self.attention_s(dec_output)
        enc_num = self.num_words_meeting
        dec_num = self.summary_length

        et = torch.zeros((batch_size, dec_num, enc_num, self.hidden_size)).cuda()
        for i in range(enc_num):
            et[:,:,i,:] = torch.tanh(Wh_ht[:,i,:].unsqueeze(1) + Ws_st[:,:,:])
        et = self.attention_v(et).squeeze(-1)
        at = self.attention_softmax(et) # [batch_size, dec_num, enc_num]
        context_vec = torch.bmm(at, enc_output) # [batch_size, dec_num, hidden_size*2]

        # Output Layer
        output = self.decoder_linear1(torch.cat((dec_output,context_vec),dim=-1))
        output = self.decoder_softmax(self.decoder_linear2(output)) # [batch_size, dec_num, vocab_size]

        # P(generating)
        Pc_ct = self.pointer_c(context_vec).squeeze(-1)
        Ps_st = self.pointer_s(dec_output).squeeze(-1)
        Px_xt = self.pointer_x(dec_embed).squeeze(-1)
        Pgen = torch.sigmoid(Pc_ct+Ps_st+Px_xt).unsqueeze(-1)

        # overall P(w)
        Pvocab = output
        # during training --- it's teacher forcing
        Ppt = torch.zeros((batch_size, dec_num, self.vocab_size)).cuda()
        for bi in range(batch_size):
            for t in range(dec_num-1):
                yt = target[bi,t+1]
                pos = (input_pgn[bi,:] == yt).float().cuda()
                pt_attn = at[bi,t,:]
                Ppt[bi,t,yt] = (pos*pt_attn).sum()
        Pw = Pgen*Pvocab + (1.0-Pgen)*Ppt

        return self.logsoftmax(Pw)
