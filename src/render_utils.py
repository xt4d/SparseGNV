import torch
import cv2
import os

import numpy as np
# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)


def obtain_lookat_up_campos(c2w):
    w2c = np.linalg.inv(c2w)
    campos = c2w[:, :3, 3]  # Not: cam pos is c2w[:3, 3]
    look_at_dir = w2c[:, 2, :3]
    look_at = campos + look_at_dir
    up = w2c[:, 1, :3]
    return look_at, up, campos


def unproject_rgbd(color, depth, intrin_depth, pose):

    height = depth.shape[0]
    width = depth.shape[1]

    py, px = torch.meshgrid(
        torch.arange(0, height, dtype=torch.float32, device=color.device),
        torch.arange(0, width, dtype=torch.float32, device=color.device)
    )

    img_xy = torch.stack([px, py], dim=-1)

    cam_xy =  img_xy * depth
    cam_xyz = torch.cat([cam_xy, depth], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrin_depth).t()

    valid = cam_xyz[..., 2] > 0

    color = color[valid, :]
    cam_xyz = cam_xyz[valid, :]

    cam_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,:1])], dim=-1)
    points_xyz = (cam_xyz.view(-1, 4) @ pose.t())[...,:3]
    points_color = color.view(-1, 3)

    return points_xyz, points_color


def build_point_cloud(scan_data, device='cpu'):

    all_verts = torch.zeros((0, 3), dtype=torch.float32, device=device)
    all_colors = torch.zeros((0, 3), dtype=torch.float32, device=device)

    for vid in scan_data['source']:

        color_img = scan_data['source'][vid]['color']
        depth_img = scan_data['source'][vid]['depth']
        pose = scan_data['source'][vid]['pose']

        intrin = scan_data['intrin_depth'][:3, :3].copy()

        verts, colors = unproject_rgbd(
            torch.tensor(color_img, device=device), 
            torch.tensor(depth_img, device=device), 
            torch.tensor(intrin, device=device), 
            torch.tensor(pose, device=device)
        )

        all_verts = torch.cat((all_verts, verts), dim=0)
        all_colors = torch.cat((all_colors, colors), dim=0)

    return all_verts, all_colors


def get_point_cloud(verts, colors):
    return Pointclouds(points=[verts], features=[colors])


def render_point_cloud(scan_data, pcds, device='cpu', bg_color=[-1., -1., -1.]):

    render_width = scan_data['render_width']
    render_height = scan_data['render_height']

    raster_settings = PointsRasterizationSettings(
        image_size=(render_height, render_width),
        radius = 0.005,
        points_per_pixel = 10
    )

    out_colors = torch.zeros((0, render_height, render_width, 3), dtype=torch.float32, device='cpu')
    
    rendered_vids = []
    for vid in scan_data['target']:

        c2w = scan_data['target'][vid]['pose'][np.newaxis, ...].astype(np.float32)

        if not np.all(np.isfinite(c2w)):
            print(vid, 'infinite')
            continue

        lookat, up, campos = obtain_lookat_up_campos(c2w)

        lookat = torch.Tensor(lookat)
        campos = torch.Tensor(campos)
        up = torch.Tensor(up)
        rotation, translate = look_at_view_transform(eye=campos, at=lookat, up=-up)

        # Create a points renderer by compositing points using an alpha compositor (nearer points
        # are weighted more heavily). 

        fov = np.arctan( float(scan_data['color_width']) * 0.5 / float(scan_data['intrin_color'][0, 0]) ) * 2 * 180 / np.pi

        cameras = FoVPerspectiveCameras(device=device, R=rotation, T=translate, fov=fov, znear=0.5)
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

        renderer = PulsarPointsRenderer(
            max_num_spheres=int(3e7),
            rasterizer=rasterizer,
            n_channels=3
            #compositor=AlphaCompositor(background_color=(255, 255, 255))
        ).to(device)

        colors = renderer(
            pcds, 
            gamma=(1e-3,), 
            mode=0,
            bg_col=torch.tensor(bg_color, dtype=torch.float32, device=device)
        )
        
        out_colors = torch.vstack((out_colors, colors.cpu()))
        rendered_vids.append(vid)

        colors = None

    return rendered_vids, out_colors


def load_data(scene_root, source_vids, target_vids, render_width, render_height, min_depth=0.3, max_depth=8.0):

    def load_rgb_image(fpath):
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.array(img, dtype=np.float32)

    def load_depth_image(fpath, depth_scale, trunc_min, trunc_max):
        img = cv2.imread(fpath, -1).astype(np.float32)
        img = img / depth_scale
        img[img > trunc_max] = 0
        img[img < trunc_min] = 0
        return img

    export_root = os.path.join(scene_root, 'exported')

    intrin_color = np.loadtxt(os.path.join(export_root, 'intrinsic', 'intrinsic_color.txt')).astype(np.float32)
    intrin_depth = np.loadtxt(os.path.join(export_root, 'intrinsic', 'intrinsic_depth.txt')).astype(np.float32)

    color_img = load_rgb_image(os.path.join(export_root, 'color', f'{source_vids[0]}.jpg'))

    scan_data = {
        'intrin_color': intrin_color,
        'intrin_depth': intrin_depth,
        'source': {},
        'target': {},
        'render_width': render_width,
        'render_height': render_height,
        'color_width': color_img.shape[1],
        'color_height': color_img.shape[0]
    }

    for vid in source_vids:

        pose = np.loadtxt(os.path.join(export_root, 'pose', f'{vid}.txt')).astype(np.float32)
        if not np.all(np.isfinite(pose)):
            continue

        color_img = load_rgb_image(os.path.join(export_root, 'color', f'{vid}.jpg'))
        depth_img = load_depth_image(os.path.join(export_root, 'depth', f'{vid}.png'), 1000, min_depth, max_depth)

        if os.path.exists(os.path.join(export_root, 'confidence', f'{vid}.png')):
            conf_img = cv2.imread(os.path.join(export_root, 'confidence', f'{vid}.png'), -1)
            depth_img[conf_img < 255] = 0

        '''resize to depth size'''
        color_img = cv2.resize(color_img, (depth_img.shape[1], depth_img.shape[0]))
        depth_img = depth_img[..., np.newaxis]

        scan_data['source'][vid] = {
            'color': color_img,
            'depth': depth_img,
            'pose': pose
        }

        scan_data['target'][vid] = {
            'pose': pose
        }

    for vid in target_vids:

        pose = np.loadtxt(
            os.path.join(
                scene_root, 
                'novel_pose', 
                f'{vid}.txt'
            )
        ).astype(np.float32)

        if not np.all(np.isfinite(pose)):
            continue

        scan_data['target'][vid] = {
            'pose': pose
        }

    return scan_data


def export_obj(fpath, xyz, rgb):

    print(os.path.dirname(fpath))
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w+") as f:
        for xyz, rgb in zip(xyz.tolist(), rgb.tolist()):
            f.write("v %f %f %f %f %f %f\n" % (xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2]))
