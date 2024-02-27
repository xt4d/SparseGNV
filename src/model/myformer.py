import torch
from .modules.scan_encoder import ScanEncoder
from .modules.decoder import CausalDecoder


class MyFormer(torch.nn.Module):

    def __init__(self, vocab_size, seq_length, hidden_dim=768):

        super().__init__()

        self.encoder = ScanEncoder(hidden_dim=hidden_dim)
        self.decoder = CausalDecoder(token_dim=vocab_size, seq_length=seq_length, bos_value=vocab_size-1, model_dim=hidden_dim)


    '''   rgb   [batch, frame, H, W, 3]      '''
    '''   pfeat [batch, frame, H, W, 3]      '''
    '''   mask  [batch, frame, H, W, 1]      '''
    '''   pose  [batch, frame, 9+3]            '''
    '''   rays  [batch, frame, H, W, 3]      '''
    '''   view_type [batch, frame] '''
    def encode(self, rgb, pfeat, mask, pose, ray, view_type):

        rgb = rgb.permute(0, 1, 4, 2, 3).contiguous()
        pfeat = pfeat.permute(0, 1, 4, 2, 3).contiguous()
        mask = mask.permute(0, 1, 4, 2, 3).contiguous()

        return self.encoder(rgb, pfeat, mask, pose[..., -3:], ray, view_type)

    
    def forward(self, rgb, pfeat, mask, pose, ray, view_type, target_code):

        cond_h = self.encode(rgb, pfeat, mask, pose, ray, view_type)
        
        ''' batch frame D -> frame batch D '''
        cond_h = cond_h.permute(1, 0, 2).contiguous()
        target_code = target_code.permute(1, 0).contiguous()

        loss = self.decoder(target_code, cond_h)

        return loss

