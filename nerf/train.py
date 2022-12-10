import torch
import numpy as np
from torch import nn, Tensor
from nerf.nerf_helper import nerf_step_sampled, get_rays
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt

# def train_step(
#     nerf, 
#     optimzer,
#     images,
#     poses,
#     int_mat,
#     threshold,
#     N_pos,
#     N_dir,
#     N_sample,
#     batch_size,
#     device
#     ):
#     index = np.random.randint(len(images))
#     image = images[index].clone().to(device)
#     pose = poses[index].clone().to(device)

#     h, w , _ = image.shape

#     optimzer.zero_grad()
    
#     pred_rgb, _ = nerf_step(
#         nerf,
#         (h,w),
#         int_mat.to(device),
#         pose,
#         threshold,
#         True,
#         N_pos,
#         N_dir,
#         N_sample,
#         batch_size
#     )

#     loss = F.mse_loss(pred_rgb, image[...,:3])
#     loss.backward()
#     optimzer.step()

def train_step_sampled(
    nerf_coarse, 
    nerf_fine,
    optimzer,
    scheduler,
    images_train,
    rays_o_list,
    rays_d_list,
    threshold,
    N_pos,
    N_dir,
    N_sample,
    N_importance,
    batch_size,
    mini_batch,
    device
    ):
    # reconstruct a fake image
    # first select 3 images

    # img1,img2,img3 = torch.randint(199,(3,))
    # idxs = torch.cat(
    #     [
    #         torch.randint(low=800*800*img1,high=800*800*(img1+1), size=(1728,)),
    #         torch.randint(low=800*800*img2,high=800*800*(img2+1), size=(1728,)),
    #         torch.randint(low=800*800*img3,high=800*800*(img3+1), size=(1728,))
    #     ],
    #     axis=0
    #     )
    # image = images_train[idxs].reshape(72,72,3).to(device)
    # rays_o = rays_o_list[idxs].reshape(72,72,3).to(device)
    # rays_d = rays_d_list[idxs].reshape(72,72,3).to(device)

    idxs = torch.randint(images_train.shape[0], (4096,))
    image = images_train[idxs].reshape(64,64,3).to(device)
    rays_o = rays_o_list[idxs].reshape(64,64,3).to(device)
    rays_d = rays_d_list[idxs].reshape(64,64,3).to(device)

    optimzer.zero_grad()
    
    coarse_out, fine_out = nerf_step_sampled(
        nerf_coarse,
        nerf_fine,
        (64,64),
        rays_o,
        rays_d,
        threshold,
        device,
        True,
        True,
        N_pos,
        N_dir,
        N_sample,
        N_importance,
        batch_size
    )
    if not nerf_fine:
        loss = F.mse_loss(fine_out[0], image[...,:3])
    else:
        loss = F.mse_loss(fine_out[0], image[...,:3])+F.mse_loss(coarse_out[0], image[...,:3])
    loss.backward()
    optimzer.step()
    scheduler.step()


def val_step_sampled(
    iter:int,
    nerf_coarse,
    nerf_fine,
    optimizer,
    val_idx,
    images,
    poses,
    int_mat,
    threshold,
    N_pos,
    N_dir,
    psnrs,
    val_iters,
    losses,
    N_samples,
    N_importance,
    checkpoint_path_coarse,
    checkpoint_path_fine,
    batch_size,
    device
    ):

    image = images[val_idx].clone().to(device)
    pose = poses[val_idx].clone()

    rays_o, rays_d = get_rays((800,800), int_mat.cpu(), pose)

    pred_rgb = torch.zeros((800,800,3)).to(device)
    depth = torch.zeros((800,800)).to(device)
    for i in range(4):
        for j in range(4):
            sub_rays_o = rays_o[i*200:(i*200)+200,j*200:(j*200)+200,:].to(device)
            sub_rays_d = rays_d[i*200:(i*200)+200,j*200:(j*200)+200,:].to(device)

            h,w = 200,200

            _, fine_out = nerf_step_sampled(
                nerf_coarse,
                nerf_fine,
                (h,w),
                sub_rays_o,
                sub_rays_d,
                threshold,
                device,
                True,
                True,
                N_pos,
                N_dir,
                N_samples,
                N_importance,
                batch_size
            )

            pred_rgb[i*200:(i*200)+200,j*200:(j*200)+200,:] = fine_out[0]
            depth[i*200:(i*200)+200,j*200:(j*200)+200] = fine_out[1]

    loss = F.mse_loss(pred_rgb, image[...,:3])
    losses.append(float(loss))
    psnr = -10.*torch.log10(loss)
    psnrs.append(psnr.item())
    val_iters.append(iter)

    # plot
    print("Iter:", iter)
    plt.figure(figsize=(13,3))
    plt.subplot(151)
    plt.imshow(pred_rgb.detach().cpu().numpy())
    plt.title(f"iter {iter}")
    plt.subplot(152)
    plt.imshow(image.detach().cpu().numpy())
    plt.title(f"val {val_idx}, ground truth")
    plt.subplot(153)
    plt.imshow((depth.detach().cpu().numpy())/640)
    plt.title(f"val {val_idx}, depth map")
    plt.subplot(154)
    plt.plot(val_iters, psnrs)
    plt.title("PSNR")
    plt.subplot(155)
    plt.plot(val_iters, losses)
    plt.title("Loss")
    plt.show()


    # save checkpoint coarse
    torch.save({
        'epoch': iter,
        'model_state_dict': nerf_coarse.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'psnrs': psnrs,
        'its': val_iters},
        checkpoint_path_coarse
        )
    
    if nerf_fine:
        # save checkpoint fine
        torch.save({
            'epoch': iter,
            'model_state_dict': nerf_fine.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'psnrs': psnrs,
            'its': val_iters},
            checkpoint_path_fine
            )


def train_sampled(
    nerf_coarse,
    nerf_fine,
    optimizer,
    scheduler,
    imgs_train,
    rays_o_list,
    rays_d_list,
    imgs_val,
    poses_val,
    val_idx,
    int_mat,
    threshold,
    N_pos,
    N_dir,
    N_sample,
    N_importance,
    checkpoint_path_coarse,
    checkpoint_path_fine,
    batch_size,
    psnrs,
    val_iters,
    losses,
    epochs,
    val_gap,
    mini_batch,
    device="cuda"
    ):

    for i in tqdm(range(epochs)):
        if i%val_gap == 0:
            with torch.no_grad():
                val_step_sampled(
                    i,
                    nerf_coarse,
                    nerf_fine,
                    optimizer,
                    val_idx,
                    imgs_val,
                    poses_val,
                    int_mat,
                    threshold,
                    N_pos,
                    N_dir,
                    psnrs,
                    val_iters,
                    losses,
                    N_sample,
                    N_importance,
                    checkpoint_path_coarse,
                    checkpoint_path_fine,
                    batch_size,
                    device
                    )
        else:
            train_step_sampled(
                nerf_coarse=nerf_coarse,
                nerf_fine=nerf_fine,
                optimzer=optimizer,
                scheduler=scheduler,
                images_train=imgs_train,
                rays_o_list=rays_o_list,
                rays_d_list=rays_d_list,
                threshold=threshold,
                N_pos=N_pos,
                N_dir=N_dir,
                N_sample=N_sample,
                N_importance=N_importance,
                batch_size=batch_size,
                mini_batch=mini_batch,
                device=device
            )

