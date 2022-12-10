import numpy as np
import torch
from PIL import Image
import os

def load_data(data_path:str, data_type:str, scale_factor: int = 8, device='cpu') -> tuple:
    assert (data_type in ("train","val","test"))

    if data_type == "train":
        start_num = '0'
    elif data_type == "val":
        start_num = '1'
    else:
        start_num = '2'
    
    images, poses = [], []

    if data_type != "test":
        # load image data
        for file in sorted(os.listdir(f"{data_path}/rgb")):
            if file.startswith(start_num):
                img = Image.open(f'{data_path}/rgb/{file}')
                if scale_factor != 1:
                    img = img.resize(
                        (int(img.width/scale_factor),
                        int(img.height/scale_factor)),
                        Image.Resampling.LANCZOS
                        )
                img = torch.Tensor(np.array(img.convert('RGB'))/255)
                images.append(img)
    
    # transfer image to tensor
        images_np = np.stack(images, axis=0)
        images = torch.from_numpy(images_np)
        images = images.to(device=device)
    
    
    # load pose data
    for file in sorted(os.listdir(f"{data_path}/pose")):
        if file.startswith(start_num):
            pose = torch.Tensor(np.loadtxt(f'{data_path}/pose/{file}'))
            # transfer the pose to right format
            pose[:,1:3] *= -1

            poses.append(pose)
    
    # transfer poses to tensor
    poses_np = np.stack(poses, axis=0)
    poses = torch.from_numpy(poses_np).to(device=device)
    
    # load intrinsics
    intrinsic_mat = np.loadtxt(f"{data_path}/intrinsics.txt")/scale_factor
    intrinsic_mat[2][2] = 1

    intrinsic_mat = torch.Tensor(intrinsic_mat).to(device=device)

    return images, poses, intrinsic_mat