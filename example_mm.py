import torch
import argparse
import numpy as np
import os
import open3d as o3d  # Added Open3D for geometric processing
from pi3.utils.basic import load_multimodal_data, write_ply
from pi3.utils.geometry import depth_edge, recover_intrinsic_from_rays_d
from pi3.models.pi3x import Pi3X

def apply_planar_projection(points_np, colors_np=None, radius=0.04, iterations=3):
    """
    Applies Moving Least Squares (MLS) Planar Projection to a point cloud array.
    """
    pcd = o3d.geometry.PointCloud()
    
    # Open3D works best with float64
    pcd.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))
    if colors_np is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors_np.astype(np.float64))

    num_points = len(points_np)
    
    for it in range(iterations):
        print(f"  -> Smoothing iteration {it + 1}/{iterations}...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
        
        normals = np.asarray(pcd.normals)
        points = np.asarray(pcd.points)
        new_points = np.copy(points)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        
        for i in range(num_points):
            [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
            if k > 3:
                neighbors = points[idx, :]
                centroid = np.mean(neighbors, axis=0)
                normal = normals[i]
                
                vector_to_point = points[i] - centroid
                distance_to_plane = np.dot(vector_to_point, normal)
                new_points[i] = points[i] - (distance_to_plane * normal)
                
        pcd.points = o3d.utility.Vector3dVector(new_points)
    
    return np.asarray(pcd.points), np.asarray(pcd.colors) if colors_np is not None else None

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    
    # parser.add_argument("--conditions_path", type=str, default='examples/room/condition.npz',
    parser.add_argument("--conditions_path", type=str, default=None,
                        help="Optional path to a .npz file containing 'poses', 'depths', 'intrinsics'.")

    parser.add_argument("--save_path", type=str, default='examples/result.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # 1. Prepare input data
    device = torch.device(args.device)

    # Load optional conditions from .npz
    poses = None
    depths = None
    intrinsics = None

    if args.conditions_path is not None and os.path.exists(args.conditions_path):
        print(f"Loading conditions from {args.conditions_path}...")
        data_npz = np.load(args.conditions_path, allow_pickle=True)

        poses = data_npz['poses']             # Expected (N, 4, 4) OpenCV camera-to-world
        depths = data_npz['depths']           # Expected (N, H, W)
        intrinsics = data_npz['intrinsics']   # Expected (N, 3, 3)

    conditions = dict(
        intrinsics=intrinsics,
        poses=poses,
        depths=depths
    )

    # Load images (Required)
    imgs, conditions = load_multimodal_data(args.data_path, conditions, interval=args.interval, device=device) 
    use_multimodal = any(v is not None for v in conditions.values())
    if not use_multimodal:
        print("No multimodal conditions found. Disable multimodal branch to reduce memory usage.")

    # 2. Prepare model
    print(f"Loading model...")
    if args.ckpt is not None:
        model = Pi3X(use_multimodal=use_multimodal).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`
        if not use_multimodal:
            model.disable_multimodal()
    model = model.to(device)

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(
                imgs=imgs, 
                **conditions
            )

    # 3.5 Recover intrinsic from rays_d
    rays_d = torch.nn.functional.normalize(res['local_points'], dim=-1)
    K = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)
    print(f"Recovered first frame intrinsic: \n{K[0, 0].cpu().numpy()}")
    if conditions['intrinsics'] is not None:
        print(f"Original first frame intrinsic: \n{conditions['intrinsics'][0, 0].cpu().numpy()}")

    # 4. process mask
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # 5. Extract, Smooth, and Save points
    print("Extracting points for geometry refinement...")
    
    # Extract raw masked points and colors to CPU numpy arrays
    raw_points_np = res['points'][0][masks].cpu().numpy()
    raw_colors_np = imgs[0].permute(0, 2, 3, 1)[masks].cpu().numpy()

    # Apply the Planar Projection logic
    print("Applying Planar Projection to collapse multi-view thickness...")
    smoothed_points_np, smoothed_colors_np = apply_planar_projection(
        points_np=raw_points_np, 
        colors_np=raw_colors_np, 
        radius=0.04, 
        iterations=3
    )

    # Convert back to PyTorch tensors so the native write_ply function works normally
    final_points_tensor = torch.from_numpy(smoothed_points_np).float()
    final_colors_tensor = torch.from_numpy(smoothed_colors_np).float()

    print(f"Saving point cloud to: {args.save_path}")
    if os.path.dirname(args.save_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        
    # Use the pipeline's native writer with the newly smoothed tensors
    write_ply(final_points_tensor, final_colors_tensor, args.save_path)
    print("Done.")
