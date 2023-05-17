import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class Quantizer(nn.Module):
    def __init__(self, num_embs, emb_dim, Beta):

        super().__init__()
   
        self.num_embs = num_embs #vocab size (K in paper)
        self.emb_dim = emb_dim #space to quantize (D in paper)
        self.Beta = Beta

        self.embedding = nn.Embedding(self.num_embs, self.emb_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embs, 1/self.num_embs)
        
    def forward(self, z_e_x):
        # z_e_x: [B, D, H, W]
        x = z_e_x.permute(0, 2, 3, 1).contiguous()
        # x: [B, H, W, D]
        x_flat = x.view(-1, self.emb_dim) # [B*H*W, D]
        # x_flat: [B*H*W, D]
        distances = (torch.sum(x_flat**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(x_flat, self.embedding.weight.t()))
        # distances: [B*H*W, K]

        e_i_stars_idx = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(e_i_stars_idx.shape[0], self.num_embs, device=z_e_x.device)
        encodings.scatter_(1, e_i_stars_idx, 1) # One hot vectorization: [B*H*W, K]
        # encodings:[B*H*W, K] * emb.weight[K, D] => quantized[B*H*W, D] => view: [B, H, W, D]
        quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)
        
        # Loss
        loss_q = F.mse_loss(quantized, x.detach())
        loss_e = F.mse_loss(quantized.detach(), x)
        loss = loss_q + self.Beta * loss_e
        
        quantized = x + (quantized - x).detach() #copy the gradient from z_q_x to z_e_x
        z_q_x = quantized.permute(0, 3, 1, 2).contiguous()
        # z_q_x: [B, D, H, W] -> similar to z_q_x

        probs = torch.mean(encodings, dim=0) #prob of selecting specific e_i
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10))) # higher perplexity is better
        # Max(perplexity) = K, provided that [K / BHW] -> 0
        
        return loss, z_q_x, perplexity, e_i_stars_idx
        
        
        

class Block(nn.Module):

    def __init__(self, n_in, n_out, n_layers, kind):

        super().__init__()
        
        if kind == 'encoder':
            first_krn, last_krn = 3, 1
        else:
            first_krn, last_krn = 1, 3
        
        n_hid = n_out // 4

        self.gain = 1 / (n_layers ** 2)
        
        self.identity_conv = nn.Conv2d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.res_block = nn.Sequential(
                                        nn.ReLU(), nn.Conv2d(n_in, n_hid, first_krn, padding='same'),
                                        nn.ReLU(), nn.Conv2d(n_hid, n_hid, 3, padding='same'),
                                        nn.ReLU(), nn.Conv2d(n_hid, n_hid, 3, padding='same'),
                                        nn.ReLU(), nn.Conv2d(n_hid, n_out, last_krn, padding='same')
                                        )

    def forward(self, x):

        return self.identity_conv(x) + self.gain * self.res_block(x)
        



class Encoder(nn.Module):

    def __init__(self, inp_ch, n_hid, block_per_group, emb_dim, group_count=4):
        '''
          @inp_ch: 3
          @n_hid: 256 in the paper        
          @block_per_group: 2
          @group_count: 4

        '''
        super().__init__()
        
        n_layers = group_count * block_per_group

        self.conv_first = nn.Conv2d(inp_ch, n_hid, 7, padding='same')
        self.group1 = nn.Sequential(OrderedDict([(f'GE1_{i+1}', Block(n_hid, n_hid, n_layers, 'encoder')) for i in range(block_per_group)]))
        self.group2 = nn.Sequential(OrderedDict([(f'GE2_{i+1}', Block(1*n_hid if i==0 else 2*n_hid, 2*n_hid, n_layers, 'encoder')) for i in range(block_per_group)]))
        self.group3 = nn.Sequential(OrderedDict([(f'GE3_{i+1}', Block(2*n_hid if i==0 else 4*n_hid, 4*n_hid, n_layers, 'encoder')) for i in range(block_per_group)]))
        self.group4 = nn.Sequential(OrderedDict([(f'GE4_{i+1}', Block(4*n_hid if i==0 else 8*n_hid, 8*n_hid, n_layers, 'encoder')) for i in range(block_per_group)]))

        self.conv_final = nn.Conv2d(8*n_hid, emb_dim, 1) 

        self.maxpool = nn.MaxPool2d(kernel_size=2)


    def forward(self, x):

        x = self.conv_first(x)
        x = self.maxpool(self.group1(x))
        x = self.maxpool(self.group2(x))
        x = self.maxpool(self.group3(x))
        x = self.group4(x)

        return self.conv_final(F.relu(x))      
        
        
        
        
class Decoder(nn.Module):

    def __init__(self, n_init, n_hid, block_per_group, emb_dim, out_ch, group_count=4):
        '''
          @n_init: 128 in the paper
          @n_hid: 256 in the paper        
          @block_per_group: 2
          @group_count: 4

        '''
        super().__init__()
        
        n_layers = group_count * block_per_group

        self.conv_first = nn.Conv2d(emb_dim, n_init, 1)
        self.group1 = nn.Sequential(OrderedDict([(f'GD1_{i+1}', Block(n_init if i==0 else 8*n_hid, 8*n_hid, n_layers, 'decoder')) for i in range(block_per_group)]))
        self.group2 = nn.Sequential(OrderedDict([(f'GD2_{i+1}', Block(8*n_hid if i==0 else 4*n_hid, 4*n_hid, n_layers, 'decoder')) for i in range(block_per_group)]))
        self.group3 = nn.Sequential(OrderedDict([(f'GD3_{i+1}', Block(4*n_hid if i==0 else 2*n_hid, 2*n_hid, n_layers, 'decoder')) for i in range(block_per_group)]))
        self.group4 = nn.Sequential(OrderedDict([(f'GD4_{i+1}', Block(2*n_hid if i==0 else 1*n_hid, 1*n_hid, n_layers, 'decoder')) for i in range(block_per_group)]))

        self.conv_final = nn.Conv2d(1*n_hid, out_ch, 1) 

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, x):

        x = self.conv_first(x)
        x = self.upsample(self.group1(x))
        x = self.upsample(self.group2(x))
        x = self.upsample(self.group3(x))
        x = self.group4(x)
        x = self.conv_final(F.relu(x))

        return x   
        



class dVAE(nn.Module):
    def __init__(self, inp_ch, n_hid, n_init, block_per_group, vocab_size, emb_dim, Beta):
      
        super().__init__()
        
        self.encoder = Encoder(inp_ch, n_hid, block_per_group, emb_dim)

        self.QNTZ = Quantizer(vocab_size, emb_dim, Beta)

        self.decoder = Decoder(n_init, n_hid, block_per_group, emb_dim, inp_ch)

    @torch.no_grad()
    def get_code_book(self, image):
        
        bs = image.size(0)
        z_e_x = self.encoder(image)
        return self.QNTZ(z_e_x)[-1].view(bs, -1) # [B, 1024]

    def forward(self, x):

        z_e_x = self.encoder(x) # [batch_size, emb_dim, 32, 32]

        loss, z_q_x, perplexity, _ = self.QNTZ(z_e_x)
        x_tilda = self.decoder(z_q_x)

        return loss, x_tilda, perplexity
        
