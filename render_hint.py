import os
import numpy as np
import argparse
import cv2
import src.render_utils as srutils


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str)
parser.add_argument('--pcd_ratio', type=float, default=0.66)
parser.add_argument('--render_width', type=int, default=640)
parser.add_argument('--render_height', type=int, default=480)
parser.add_argument('--seed', default=None)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--export_root', type=str, default=None)

args = parser.parse_args()

if args.seed is not None:
    np.random.seed(int(args.seed))

data_root = args.data_root

export_root = os.path.join(data_root, 'render_hint') if args.export_root is None else args.export_root
os.makedirs(export_root, exist_ok=True)

with open(os.path.join(data_root, 'obs_vids.txt'), 'r') as fin:
    obs_vids = fin.readline().strip().split(',')

with open(os.path.join(data_root, 'novel_vids.txt'), 'r') as fin:
    novel_vids = fin.readline().strip().split(',')
 
scan_data = srutils.load_data(data_root, obs_vids, novel_vids, args.render_width, args.render_height)

pcd_verts, pcd_colors = srutils.build_point_cloud(scan_data, device=args.device)
print(len(obs_vids), len(scan_data['source']), len(novel_vids), pcd_verts.shape)

sel = np.random.choice(pcd_verts.shape[0], size=int(pcd_verts.shape[0] * args.pcd_ratio), replace=False)
pcd_verts = pcd_verts[sel, ...]
pcd_colors = pcd_colors[sel, ...]

print('After sampling', pcd_verts.shape)

pcds = srutils.get_point_cloud(pcd_verts, pcd_colors)

out_vids, out_colors = srutils.render_point_cloud(scan_data, pcds, device=args.device)

for vid, color in zip(out_vids, out_colors):

    color = color.cpu().numpy()

    mask = (color[:, :, 0] < 0) * 255

    color = np.clip(color, 0, 255)

    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(export_root, f'{vid}_color.jpg'), color)
    cv2.imwrite(os.path.join(export_root, f'{vid}_mask.jpg'), mask)
