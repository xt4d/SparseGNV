import numpy as np
import torch
import torch.nn as nn
from .layers import RayEncoder, Transformer


class SRTConvBlock(nn.Module):
    def __init__(self, idim, hdim=None, odim=None):
        super().__init__()
        if hdim is None:
            hdim = idim

        if odim is None:
            odim = 2 * hdim

        conv_kwargs = {'bias': False, 'kernel_size': 3, 'padding': 1}
        self.layers = nn.Sequential(
            nn.Conv2d(idim, hdim, stride=1, **conv_kwargs),
            nn.ReLU(),
            nn.Conv2d(hdim, odim, stride=2, **conv_kwargs),
            nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class ScanEncoder(nn.Module):
    """
    Modified based on Scene Representation Transformer Encoder with the improvements from Appendix A.4 in the OSRT paper.
    """
    def __init__(self, hidden_dim, num_conv_blocks=3, num_att_blocks=6, pos_start_octave=0):
        super().__init__()
        self.ray_encoder = RayEncoder(pos_octaves=3, pos_start_octave=pos_start_octave,
                                      ray_octaves=3)

        conv_blocks = [SRTConvBlock(idim=36+3+3+1, hdim=48, odim=48)]
        cur_hdim = 48 #192
        for i in range(1, num_conv_blocks):
            conv_blocks.append(SRTConvBlock(idim=cur_hdim, odim=cur_hdim))
            #cur_hdim *= 2

        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.to_patch_sequence = nn.Unfold(kernel_size=4, stride=4)
        # out: [B*N, 8x8, 768]

        self.patch_mlp = nn.Linear(768, hidden_dim)

        self.view_type_embedding = torch.nn.Embedding(3, hidden_dim)

        self.transformer = Transformer(hidden_dim, depth=num_att_blocks, heads=8, dim_head=8,
                                       mlp_dim=1536, dropout=0.1, selfatt=True)


    def forward(self, rgb, pfeat, mask, camera_pos, rays, view_type):
        """
        Args:
            rgb: [batch_size, frame, 3, height, width]. Assume the first image is canonical.
            pfeat: [batch_size, frame, 3, height, width]. 
            mask: [batch_size, frame, 1, height, width].
            camera_pos: [batch_size, frame, 3]
            rays: [batch_size, frame, height, width, 3]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
        """

        batch_size, num_images = rgb.shape[:2]

        x = torch.cat([rgb.flatten(0, 1), pfeat.flatten(0, 1), mask.flatten(0, 1)], dim=1)
        camera_pos = camera_pos.flatten(0, 1)
        rays = rays.flatten(0, 1)

        ray_enc = self.ray_encoder(camera_pos, rays)

        x = torch.cat((x, ray_enc), 1)

        x = self.conv_blocks(x)
        x = self.to_patch_sequence(x).transpose(2, 1)
        x = self.patch_mlp(x)
        #x = x.flatten(2, 3).permute(0, 2, 1)

        '''batch_size*frame, patches, hidden_dim'''
        patches_per_image, channels_per_patch = x.shape[1:]
        x = x.reshape(batch_size, num_images * patches_per_image, channels_per_patch)

        '''vemb [batch_size, frame, patches, hidden_dim]'''
        vemb = self.view_type_embedding(view_type)[:, :, None, :].repeat(1, 1, patches_per_image, 1)
        vemb = vemb.flatten(1, 2)

        x = self.transformer(x + vemb)

        return x


if __name__ == '__main__':

    device = 'cuda:0'
    encoder = ScanEncoder(768).to(device)

    rgb = torch.randn((4, 16, 3, 256, 256), device=device)
    pfeat = torch.randn((4, 16, 3, 256, 256), device=device)
    mask = torch.randn((4, 16, 1, 256, 256), device=device)
    pos = torch.randn((4, 16, 3), device=device)
    rays = torch.randn((4, 16, 256, 256, 3), device=device)
    view_type = torch.randint(0, 3, size=(4, 16), dtype=torch.long, device=device)

    y = encoder(rgb, pfeat, mask, pos, rays, view_type)
