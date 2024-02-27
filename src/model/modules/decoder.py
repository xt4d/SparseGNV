import torch


class CausalDecoder(torch.nn.Module):

    def __init__(self, token_dim, seq_length, bos_value = 0, model_dim = 64, head_num = 8, layer_num = 6):

        super(CausalDecoder, self).__init__()

        self.seq_length = seq_length
        self.token_dim = token_dim
        self.head_num = head_num
        self.model_dim = model_dim
        self.layer_num = layer_num
        self.bos_value = bos_value

        self.pos_emb = torch.nn.Parameter(torch.randn(seq_length, model_dim))
        self.emb_layer = torch.nn.Embedding(token_dim, model_dim)

        decoder_layer = torch.nn.TransformerDecoderLayer(d_model = model_dim, nhead = head_num)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers = layer_num)

        self.logits_head = torch.nn.Linear(model_dim, token_dim)


    def decode(self, x, cond_h, add_bos=True):

        qlen = x.shape[0]
        bsz = x.shape[1]

        attn_mask = CausalDecoder.generate_square_subsequent_mask(qlen).to(x.device)

        x_with_bos = x

        if add_bos:
            bos_token = torch.ones((1, bsz), dtype=x.dtype, device=x.device) * self.bos_value
            x_with_bos = torch.cat((bos_token, x), dim=0)[:-1]

        e = self.emb_layer(x_with_bos) + self.pos_emb[:qlen, :].unsqueeze(1)

        h = self.decoder(e, cond_h, tgt_mask = attn_mask)

        return h


    def category_loss(self, h, x):
        
        tot = x.shape[0]*x.shape[1]
    
        logits = self.logits_head(h)
        
        loss_ce = torch.nn.functional.cross_entropy(logits.reshape((tot, -1)), x.reshape(tot))
        
        return loss_ce
        
    
    def forward(self, x, cond_h):

        dec_h = self.decode(x, cond_h)

        loss = self.category_loss(dec_h, x)

        return loss


    @staticmethod
    def generate_square_subsequent_mask(sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
