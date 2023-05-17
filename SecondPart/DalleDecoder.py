import torch 
from torch import nn
import torch.nn.functional as F
import math

class TextTokenEmbedding(nn.Module):

    def __init__(self, text_vocab_size, text_seq_len, d_model):

        '''
          @text_vocab_size = len(text_vocab): 16384 in the paper

          @text_seq_len: 256 in the paper -> is used here to consider <pad_i> token
          for each text position. So, we do not consider zero-padding. 

          @d_model: 512 in the paper

          It tokenizes "<bos>: Beginning_Of_Sentence Token" as index 0.

          Example: text = 'I am happy' -> [1, 29, 53, <pad>: 16387, <pad>: 16388, ..., <pad>: 16639] ->
                                          [<bos>:0, 1, 29, 53, 16387, 16388, ..., 16639] (len: 257)

          All these preprocessings take place outside of this class when preparing the data.
        '''

        super().__init__()

        self.embed = nn.Embedding(text_vocab_size + text_seq_len, d_model)

    def forward(self, x):
    
        # x: [batch_size, seq_len_TXT + 1: 257 in the paper]
        return self.embed(x) # [batch_size, seq_len_TXT + 1: 257, d_model]
        



class ImageTokenEmbedding(nn.Module):

    def __init__(self, image_vocab_size, d_model):

        '''
          @image_vocab_size = len(image_vocab): 8192 in the paper

          @d_model: 512 in the paper
        '''

        super().__init__()

        self.embed = nn.Embedding(image_vocab_size, d_model)

    def forward(self, x):
    
        # x: [batch_size, seq_len_IMG: 1024 in the paper]
        return self.embed(x) # [batch_size, seq_len_IMG, d_model]
        
        
        
        
class TextTokenPositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len_TXT):

        '''
          @seq_len_TXT: 256 + 1 = 257 in the paper

          @d_model: 512 in the paper
        '''

        super().__init__()

        self.d_model = d_model
        positional_emb = torch.zeros(seq_len_TXT, d_model)
        # positional_emb: [257, 512]

        for pos in range(seq_len_TXT):
            for i in range(0, d_model, 2):
                positional_emb[pos, i] = math.sin(pos / (10000 ** (i/d_model)))
                positional_emb[pos, i + 1] = math.cos(pos / (10000 ** (i/d_model)))
                
        self.register_buffer('positional_emb', positional_emb)
        self.positional_emb.requires_grad = False

    def forward(self, x):
    
        # x: [batch_size, seq_len_TXT: 257, d_model: 512]
        x = x * math.sqrt(self.d_model)

        _, seq_len, _ = x.size() # seq_len: 257 in the paper
        x = x + self.positional_emb
        # self.positional_emb: [seq_len_TXT: 257, d_model: 512]
        # x:                   [batch_size, seq_len_TXT: 257, d_model: 512] 

        return x # [batch_size, seq_len_TXT: 257, d_model: 512]
        
        
        
        
class ImageTokenPositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len_IMG):

        '''
          @seq_len_IMG: 32 * 32 = 1024 in the paper

          @d_model: 512 in the paper
        '''

        super().__init__()

        self.d_model = d_model
        positional_emb = torch.zeros(seq_len_IMG, d_model)
        # positional_emb: [1024, 512]

        for pos in range(seq_len_IMG):
            for i in range(0, d_model, 2):
                positional_emb[pos, i] = math.sin(pos / (10000 ** (i/d_model)))
                positional_emb[pos, i + 1] = math.cos(pos / (10000 ** (i/d_model)))
                
        self.register_buffer('positional_emb', positional_emb)
        self.positional_emb.requires_grad = False

    def forward(self, x):
    
        # x: [batch_size, seq_len_IMG: 1024, d_model: 512]
        x = x * math.sqrt(self.d_model)

        _, seq_len, _ = x.size() # seq_len: 1024 in the paper
        x = x + self.positional_emb
        # self.positional_emb: [seq_len_IMG: 1024, d_model: 512]
        # x:                   [batch_size, seq_len_IMG: 1024, d_model: 512] 

        return x # [batch_size, seq_len_IMG: 1024, d_model: 512]
        
        
        
        
def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.permute(0, 1, 3, 2)) /  math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)

    return output




class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model, dropout = 0.1):
    
        super().__init__()
        
        self.N = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):

        batch_size = q.size(0)
                
        k = self.k_linear(k).view(batch_size, -1, self.N, self.d_k).permute(0, 2, 1, 3)
        q = self.q_linear(q).view(batch_size, -1, self.N, self.d_k).permute(0, 2, 1, 3)
        v = self.v_linear(v).view(batch_size, -1, self.N, self.d_k).permute(0, 2, 1, 3)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        concat = scores.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(concat)
    
        return output # [batch_size, 1280, 512]
        
        
        
        
class Norm(nn.Module):

    def __init__(self, d_model, eps = 1e-6):
    
        super().__init__()
    
        self.d_model = d_model
        self.eps = eps

        self.Gamma = nn.Parameter(torch.ones(self.d_model)) #learnable
        self.Beta = nn.Parameter(torch.zeros(self.d_model)) #learnable
        
    def forward(self, x):
      
        mio = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = (x - mio) / (torch.sqrt(var + self.eps))
        y = self.Gamma * x_hat + self.Beta
        return y
 
 
 
        
class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout = 0.1):
    
        super().__init__() 

        self.lin1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(d_ff, d_model)

    def forward(self, x):

        x = self.dropout(F.relu(self.lin1(x)))
        x = self.lin2(x)
        return x
      
      
      
      
class DecoderOnlyLayer(nn.Module):

    def __init__(self, d_model, heads, d_ff, dropout=0.1):

        super().__init__()

        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.attention = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.drp1 = nn.Dropout(dropout)
        self.drp2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):

        # x: [batch_size, text_seq_len + image_seq_len: 256 + 1024: 1280, d_model: 512]
        x_copied = x
        x = self.attention(x, x, x, mask) # Attention
        x = self.norm1(x_copied + self.drp1(x)) # Add & Norm
        
        x_copied = x
        x = self.ff(x) # Feed forward
        x = self.norm2(x_copied + self.drp2(x)) # Add & Norm
        # x: [batch_size, 1280, d_model: 512]
        return x
      



class DecoderOnly(nn.Module):

    def __init__(self, d_model, N, heads, d_ff):

        super().__init__()

        self.N = N # how many layers
        self.layers = nn.ModuleList([DecoderOnlyLayer(d_model, heads, d_ff) for _ in range(N)])

    def forward(self, embedded_src, mask=None):
      
        # embedded_src: [batch_size, 1280, 512]
        x = embedded_src
        for i in range(self.N):
            x = self.layers[i](x, mask)
        # x: [batch_size, 1280, 512]
        return x
        
        
        
        
class DecoderOnlyTransformer(nn.Module):

    def __init__(self, text_vocab_size, text_seq_len, image_vocab_size, image_seq_len, d_model, N, heads, d_ff):
        '''
          text_vocab_size: 16384 in the paper
          text_seq_len: 256 in the paper
          image_vocab_size: 8192 in the paper
          image_seq_len: 1024 in the paper
          d_model: 512 in the paper
  
        '''
        super().__init__()

        self.len_stream = text_seq_len + image_seq_len

        self.text_emb = TextTokenEmbedding(text_vocab_size, text_seq_len, d_model)
        self.image_emb = ImageTokenEmbedding(image_vocab_size, d_model)

        self.text_pos_enc = TextTokenPositionalEncoding(d_model, text_seq_len + 1)
        self.image_pos_enc = ImageTokenPositionalEncoding(d_model, image_seq_len)

        self.decoder_only = DecoderOnly(d_model, N, heads, d_ff)

        self.final_fc = nn.Linear(d_model, text_vocab_size + text_seq_len + image_vocab_size)
        # text_vocab_size + text_seq_len: for text and its <pad> tokens
        # image_vocab_size: for image

    def forward(self, text_tokens_with_bos_and_pads, image_tokens):
        
        text_tokens_in_stream = self.text_pos_enc(self.text_emb(text_tokens_with_bos_and_pads))
        # text_tokens_in_stream: [batch_size, 257, 512]

        image_tokens_in_stream = self.image_pos_enc(self.image_emb(image_tokens))
        # image_tokens_in_stream: [batch_size, 1024, 512]

        stream = torch.cat((text_tokens_in_stream, image_tokens_in_stream), dim=1)
        len_stream = stream.size(1)
        
        if len_stream > self.len_stream: 
            stream = stream[:, :-1, :]
        # stream: [batch_size, 1280, 512]

        decoded_stream = self.decoder_only(stream)
        # decoded_stream: [batch_size, 1280, 512]

        out = self.final_fc(decoded_stream)
        # out: [batch_size, 1280, 16384 + 256 + 8192]

        return out
