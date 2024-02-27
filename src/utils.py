import importlib
import torch
import numpy as np


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def sample_sequence(model, seq_x, seq_h, prompt_length, steps, temperature=1.0, top_k=3, do_sample=False):
    bsz, seql = seq_x.shape[0], seq_x.shape[1]
    generated = seq_x[:, :prompt_length].transpose(1, 0).contiguous()
    with torch.no_grad():
        for _ in range(steps):
            roll_h = model.decoder.decode(generated, seq_h, add_bos=False)
            pred = model.decoder.logits_head(roll_h)
            next_token_logits = pred[-1, :, :]
            filtered_logits = top_k_logits(next_token_logits, k=top_k)
            if do_sample:
                next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            else:
                __, next_token = torch.topk(torch.softmax(filtered_logits, dim=-1), k=1, dim=-1)
            generated = torch.cat((generated, next_token.reshape(1, bsz)), dim=0)

    generated = generated.transpose(1, 0).contiguous()
    return generated[:, 1:].cpu().tolist(), roll_h


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


TINY_NUMBER = 1e-6

def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(np.clip(np.sum(vec1_unit*vec2_unit, axis=-1), -1.0, 1.0))
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    '''
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))


def get_nearest_pose_ids(tar_pose, ref_poses, num_select, tar_id=-1, angular_dist_method='vector',
                         scene_center=(0, 0, 0)):
    '''
    Args:
        tar_pose: target pose [3, 3]
        ref_poses: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams-1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
    elif angular_dist_method == 'vector':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == 'dist':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]
    return selected_ids


def pose2vector(pose):
    return np.concatenate((pose[:3, :3].reshape(9), pose[:3, 3]))


def get_rel_pose(c2w, anchor_c2w):

    if (not np.all(np.isfinite(c2w))) or (not np.all(np.isfinite(anchor_c2w))):
        pose = np.eye(4, dtype=np.float32)
        pose[3, 3] = 0.0
        return pose 

    rel_rot = c2w[:3, :3] @ anchor_c2w[:3, :3].T
    rel_tsl = -rel_rot @ anchor_c2w[:3, -1] + c2w[:3, -1]
    pose = np.zeros((4, 4), dtype=np.float32)
    pose[:3, :3] = rel_rot
    pose[:3, -1] = rel_tsl

    return pose


def transform_points(points, transform, translate=True):
    """ Apply linear transform to a np array of points.
    Args:
        points (np array [..., 3]): Points to transform.
        transform (np array [3, 4] or [4, 4]): Linear map.
        translate (bool): If false, do not apply translation component of transform.
    Returns:
        transformed points (np array [..., 3])
    """
    # Append ones or zeros to get homogenous coordinates
    if translate:
        constant_term = np.ones_like(points[..., :1])
    else:
        constant_term = np.zeros_like(points[..., :1])
    points = np.concatenate((points, constant_term), axis=-1)

    points = np.einsum('nm,...m->...n', transform, points)
    return points[..., :3]


def get_rays(width, height, intrin, pose):

    xmap = np.linspace(-1, 1, width)
    ymap = np.linspace(-1, 1, height)
    xmap, ymap = np.meshgrid(xmap, ymap)

    rays = np.stack((xmap, ymap, np.ones_like(xmap)), -1)

    rays = transform_points(
        rays,
        pose.T @ intrin.T,
        translate=False
    )
    
    rays = rays[..., :3]
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    return rays