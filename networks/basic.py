# ====================================================
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : basic.py
# ====================================================
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import pickle as pkl
import math
import timm

class Pos_embd(nn.Module):
    def __init__(self,num_token,dim,dropout):
        super(Pos_embd,self).__init__()
        self.pos_drop = nn.Dropout(p=dropout)
        self.pos_embed = nn.Parameter(torch.randn(1, num_token, dim) * .02)
    def forward(self,x):
        x = x + self.pos_embed
        return self.pos_drop(x)



class AttHierarchicalGround(nn.Module):

    def __init__(self, input_size, hidden_size, visual_dim, word_dim, num_layers=1):
        super(AttHierarchicalGround, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_dim = hidden_size // 2


        self.num_layers = num_layers
        self.word_dim = word_dim

        self.max_seg_len = 12
        dropout = 0.1

        self.embedding_word = nn.Sequential(nn.Linear(word_dim, self.embed_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))
        self.embedding_visual = nn.Sequential(nn.Linear(visual_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))
        
        # affine transformation for lstm hidden state
        self.linear1 = nn.Linear(hidden_size*2, hidden_size)

        # affine transformation for context
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)

        self.transform_visual = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


        ###########################################################
        # RNN Version
        #self.within_seg_rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.seg_rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        ############################################################

        ##########################################
        # Transformer version
        input_size = hidden_size
        #self.position = EncoderPositionalEncoding(d_model=input_size,max_len=120)
        within_encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, dim_feedforward=hidden_size, nhead=8, batch_first=True)
        # within_encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, dim_feedforward=hidden_size, nhead=8)


        self.within_seg_transformer = nn.TransformerEncoder(within_encoder_layer, num_layers=1)
        #self.pos_embd = Pos_embd(num_token=120,dim=512,dropout=0.1)
        #self.attn_block = timm.models.vision_transformer.Block(dim=512,num_heads=8,attn_drop=0.1)
        #encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, dim_feedforward=hidden_size, nhead=8, batch_first=True)
        #self.seg_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        ##########################################

    def soft_attention(self, input, context):
        """
        compute temporal self-attention
        :param input:  (batch_size, seq_len, feat_dim)
        :param context: (batch_size, feat_dim)
        :return: vfeat: (batch_size, feat_dim), beta
        """
        batch_size, seq_len, feat_dim = input.size()
        
        ####################################################
        # RNN Version
        context = context.unsqueeze(1).repeat(1, seq_len, 1)
        ####################################################

        ##################################################
        # Transformer Version
        #context = context.repeat_interleave(12, dim=1)
        ##################################################

        inputs = torch.cat((input, context), 2).view(-1, feat_dim*2)

        o = self.linear2(torch.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        beta = self.softmax(e)
        vfeat = torch.bmm(beta.unsqueeze(1), input).squeeze(1)

        return vfeat, beta


    def forward(self, videos, relation_text, mode='train'):
        """
        Without participation of textual relation, to warm-up the decoder only
        """

        frame_count = videos.shape[1]

        max_seg_num = int(frame_count / self.max_seg_len)

        ori_x = self.embedding_visual(videos).sum(dim=2).squeeze()
        
        x_trans = self.transform_visual(ori_x)

        #x_trans = self.position(x_trans)
        within_seg_rnn_out = self.within_seg_transformer(x_trans)
        #x_trans = self.pos_embd(x_trans)
        #within_seg_rnn_out = self.attn_block(x_trans)

        idx = np.round(np.linspace(self.max_seg_len-1, frame_count-1, max_seg_num)).astype('int')

        seg_rnn_input = within_seg_rnn_out[:,idx,:]
        
        
        #hidden = self.seg_transformer(seg_rnn_input)
        seg_out, hidden = self.seg_rnn(seg_rnn_input)
        
        output,_ = self.soft_attention(within_seg_rnn_out, hidden[0].squeeze(0))
        #output,_ = self.soft_attention(within_seg_rnn_out, hidden)

        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=10):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.MAX_RELATION_LENGTH = 9

        embed_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        #self.position = DecoderPositionalEncoding(d_model=hidden_size,max_len=10)
        #transformer_layer = nn.TransformerDecoderLayer(d_model=embed_size, dim_feedforward=hidden_size, nhead=8, batch_first=True)
        #self.transformer = nn.TransformerDecoder(transformer_layer, num_layers=1)

        self.linear = nn.Linear(hidden_size, vocab_size)
        #self.max_seq_length = max_seq_length
        #self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, video_out, video_hidden, relations, lengths):
        """
        Decode relation attended video feature and reconstruct the relation.
        :param video_out: (batch, seq_len, dim_hidden * num_directions)
        :param video_hidden: (num_layers * num_directions, batch_size, dim_hidden)
        :param relations:
        :param lengths:
        :return:
        """

        #batch, length = relations.shape
        #if length<self.MAX_RELATION_LENGTH:
        #    relations = torch.cat(
        #            (relations,
        #            torch.zeros([batch,self.MAX_RELATION_LENGTH-length]).int().to(relations.device)),
        #            dim=1
        #            )
        embeddings = self.embed(relations)
        #batch_size, seq_len, _ = embeddings.size()


        embeddings = torch.cat((video_out.unsqueeze(1), embeddings), 1)

        #max_len = self.MAX_RELATION_LENGTH
        #padding_mask = (torch.arange(max_len).expand(len(lengths), max_len) < torch.tensor(lengths).unsqueeze(1)).to(video_hidden.device)
        #print("padding masks:",padding_mask)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        #print("video hidden:",video_hidden.shape)

        hiddens, _ = self.lstm(packed, video_hidden)
        outputs = self.linear(hiddens[0])

        #max_len = self.MAX_RELATION_LENGTH+1
        #print("embd:",embeddings.shape)
        #print("video:",video_out.shape)

        #ahead_mask = torch.triu(torch.ones(max_len,max_len),diagonal=1).bool().to(video_hidden.device)
        #print("ahead mask:",ahead_mask)
        #hiddens = self.transformer(embeddings,video_hidden,tgt_mask=ahead_mask)

        #hiddens = self.transformer(embeddings,video_hidden)
        #outputs = self.softmax(self.linear(hiddens))
        #print("outputs:",outputs)
        return outputs

    def sample(self, video_out, states=None):
        """reconstruct the relation using greedy search"""
        sampled_ids = []
        inputs = video_out.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        sampled_ids = torch.stack(sampled_ids, 1)
