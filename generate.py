import os
import argparse
import torch
import torchvision.utils as vutils

from src.data.loader import ScanLoader
from src.utils import get_obj_from_str, sample_sequence
from src.vq_utils import load_vqmodel, decode_to_img
from src.model.myformer import MyFormer


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str)
parser.add_argument('--ckpt_path', type=str, default='ckpts/generator.pt')
parser.add_argument("--vqmodel_config", type=str, default="config/realestate_vqmodel.yaml")
parser.add_argument('--output_root', type=str, default='outputs')

args = parser.parse_args()

device = 'cuda:0'
vocab_size = 16384 + 1

vq_model = load_vqmodel(args.vqmodel_config, device)
vq_model.eval()

model = MyFormer(vocab_size, 16*16).to(device)

ckpt = torch.load(args.ckpt_path, map_location=device)
model.load_state_dict(ckpt['state_dict'], strict=False)

model.eval()

loader = ScanLoader(
    args.data_root, 
    context_num = 16,
    context_obs_num = 4, 
    device = device
)

os.makedirs(args.output_root, exist_ok=True)

scene_root = os.path.join(
    args.output_root, 
    os.path.basename(args.data_root)
)

os.makedirs(scene_root, exist_ok=True)
os.makedirs(os.path.join(scene_root, 'output'), exist_ok=True)

with torch.no_grad():

    for target_vid in loader.novel_vids:

        print(target_vid)

        batch = loader.get_context_and_query(target_vid)

        ctx_rgb = (batch['ctx_rgb'].permute(0, 3, 1, 2) + 1)/2
        ctx_mask = (batch['ctx_mask'].permute(0, 3, 1, 2) + 1)/2
        ctx_hint = (batch['ctx_hint'].permute(0, 3, 1, 2) + 1)/2
        ctx_view_type = torch.ones(ctx_rgb.shape).to(batch['ctx_view_type']) * (batch['ctx_view_type'][..., None, None, None] * 0.5)

        #query_rgb_gt = (batch['tar_rgb_gt'].permute(0, 3, 1, 2) + 1)/2
        query_rgb = (batch['tar_rgb'].permute(0, 3, 1, 2) + 1)/2
        query_mask = (batch['tar_mask'].permute(0, 3, 1, 2) + 1)/2
        query_hint = (batch['tar_hint'].permute(0, 3, 1, 2) + 1)/2
        query_view_type = torch.ones(query_rgb.shape).to(batch['tar_view_type']) * (batch['tar_view_type'][..., None, None, None] * 0.5)

        analysis_root = os.path.join(scene_root, 'analysis', str(target_vid))
        os.makedirs(analysis_root, exist_ok=True)

        vutils.save_image(ctx_rgb, os.path.join(analysis_root, 'ctx_rgb.jpg'), normalize=True)
        vutils.save_image(ctx_mask, os.path.join(analysis_root, 'ctx_mask.jpg'), normalize=True)
        vutils.save_image(ctx_hint, os.path.join(analysis_root, 'ctx_hint.jpg'), normalize=True)
        vutils.save_image(ctx_view_type, os.path.join(analysis_root, 'ctx_view_type.jpg'), normalize=True)

        #vutils.save_image(query_rgb_gt, os.path.join(analysis_root, 'gt_rgb.jpg'), normalize=True)
        vutils.save_image(query_rgb, os.path.join(analysis_root, 'query_rgb.jpg'), normalize=True)
        vutils.save_image(query_mask, os.path.join(analysis_root,'query_mask.jpg'), normalize=True)
        vutils.save_image(query_hint, os.path.join(analysis_root, 'query_hint.jpg'), normalize=True)
        vutils.save_image(query_view_type, os.path.join(analysis_root, 'query_view_type.jpg'), normalize=True)

        for key in batch:
            batch[key] = batch[key][None, ...].to(device)

        inp_rgb, \
        inp_hint, \
        inp_mask, \
        inp_pose, \
        inp_rays, \
        inp_view_type = loader.make_model_input(batch, 0)

        cond_h = model.encode(
            inp_rgb, 
            inp_hint, 
            inp_mask, 
            inp_pose, 
            inp_rays, 
            inp_view_type
        )

        bsz = cond_h.shape[0]

        cond_h = cond_h.permute(1, 0, 2).contiguous()

        seq_x = torch.ones((bsz, 1), dtype=torch.long, device=device) * (vocab_size-1)
        generated, roll_h = sample_sequence(model, seq_x, cond_h, 1, 16*16, do_sample=False)

        indcs = torch.tensor(generated, dtype=torch.long).to(device)
        dec = decode_to_img(vq_model, indcs, [bsz, 256, 16, 16])

        dec = (dec + 1) / 2 
        vutils.save_image(dec.cpu(), os.path.join(analysis_root, 'decode_rgb.jpg'), normalize=True)

        dec = torch.nn.functional.interpolate(dec, size=[480, 640], mode='bicubic')
        vutils.save_image(dec.cpu(), os.path.join(scene_root, 'output', f'{target_vid}.jpg'), normalize=True)
