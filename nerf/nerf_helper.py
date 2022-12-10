import torch
from torch import Tensor,nn
import torch.nn.functional as F


def get_rays(img_shape: tuple, int_mat: Tensor, pose: Tensor):
    h, w= img_shape
    
    i, j = torch.meshgrid(
        torch.linspace(0, w-1, w),
        torch.linspace(0, h-1, h),
        indexing="xy"
    )
    i, j = i.to(int_mat), j.to(int_mat)

    # change pixel to camera
    fx, fy, cx, cy = int_mat[0][0], int_mat[1][1], int_mat[0][2], int_mat[1][2]

    rays_d = torch.stack(
        [ (i-cx)/fx,-(j-cy)/fy, -torch.ones_like(i)], dim=-1
    )

    # change camera to world
    rays_d = torch.sum(rays_d[..., None, :]*pose[:3, :3], dim=-1)

    rays_d = rays_d/torch.linalg.norm(rays_d, dim=-1, keepdim=True)


    # origin of the rays
    rays_o = pose[:3, -1].expand(rays_d.shape)


    return rays_o, rays_d



def qurey(rays_o, rays_d, threshold: tuple, sample_num: int):
    near, far = threshold

    depth_values = torch.linspace(near, far, sample_num).to(rays_o)
    noise_shape = list(rays_o.shape[:-1])+[sample_num]
    depth_values = depth_values + \
        torch.rand(noise_shape).to(rays_o)*(far-near)/sample_num
    
    query_points = rays_o[...,None,:] + rays_d[...,None,:]*depth_values[...,:,None]

    return query_points, depth_values

def sample_pdf(
  bins: torch.Tensor,
  weights: torch.Tensor,
  n_samples: int,
  perturb: bool = False
) -> torch.Tensor:


    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True)


    cdf = torch.cumsum(pdf, dim=-1) 
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) 

    if not perturb:
        u = torch.linspace(0., 1., n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]) 
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device) 

    u = u.contiguous() 
    inds = torch.searchsorted(cdf, u, right=True) 

    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1) 

    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                        index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                        index=inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples 

def sample_hierarchical(
  rays_o: torch.Tensor,
  rays_d: torch.Tensor,
  depth_values: torch.Tensor,
  weights: torch.Tensor,
  n_samples: int,
  perturb: bool = False
):

    depth_values_mid = .5 * (depth_values[..., 1:] + depth_values[..., :-1])
    new_z_samples = sample_pdf(depth_values_mid, weights[..., 1:-1], n_samples,
                            perturb=perturb)
    new_z_samples = new_z_samples.detach()

    depth_values_combined, _ = torch.sort(torch.cat([depth_values, new_z_samples], dim=-1), dim=-1)
    pts1 = rays_o[..., None, :] + rays_d[..., None, :] * depth_values_combined[..., :, None]  # [N_rays, N_samples + n_samples, 3]
    pts2 = rays_o[..., None, :] + rays_d[..., None, :] * new_z_samples[..., :, None]
    return pts1, depth_values_combined, pts2, new_z_samples


def render(nerf_out, depth_values):
    # normalize the output
    alpha_maps = nerf_out[...,3]
    rgb_maps = nerf_out[...,:3]

    # compute span for each points by diff
    depth_spans = depth_values[...,1:]-depth_values[...,:-1]
    depth_spans = torch.cat([
        depth_spans,
        torch.Tensor([1e10]).to(depth_spans).expand_as(depth_values[...,:1])
    ], dim=-1)

    # add noise (this may help?)
    # noise = torch.randn(alpha_maps.shape, device=alpha_maps.device)

    # compute the weights for each point in pixel by alpha and depth span
    alpha_maps = 1.-torch.exp(-(alpha_maps)*depth_spans)

    transmit_acc = torch.cumprod(1 - alpha_maps + 1e-10, dim=-1)
    transmit_acc = transmit_acc.roll(shifts=1, dims=-1)
    transmit_acc[..., 0] = 1
    weights = alpha_maps * transmit_acc

    # get results
    rgb_img = (rgb_maps*weights.unsqueeze(-1)).sum(dim=-2)
    depth_img = (depth_values*weights).sum(dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    # for white bg
    rgb_img = rgb_img + (1.-acc_map[..., None])

    return rgb_img, depth_img, acc_map, weights



def encode_pos(x, L):
    pe = []
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            pe.append(fn(2.**i * x))
    return torch.cat(pe, -1)


def nerf_step(
    nerf: nn.Module,
    img_shape: tuple,
    int_mat: Tensor,
    pose: Tensor,
    threshold: tuple,
    pos = True,
    N_pos = 10,
    N_dir = 4,
    N_sample:int = 64,
    batch_size=32
    ):

    # get rays of all pixels of the given image
    rays_o, rays_d = get_rays(img_shape, int_mat, pose)

    # Sample points along each ray
    samples, depth_values = qurey(rays_o, rays_d, threshold, N_sample)

    # flatten the samples
    samples = samples.reshape(-1,3)

    # encode points with positions
    if pos:
        enc_samples = encode_pos(samples, N_pos)
        enc_dirs = encode_pos(rays_d.reshape(-1,3), N_dir)
    
    # train each pixel
    n = len(enc_dirs)
    output_list = []
    for i in range(0,n,batch_size):
        if (i + batch_size) > n:
            samples_cropped = enc_samples[N_sample*i:]
            dirs_cropped = enc_dirs[i:].repeat(N_sample,1)
        else:
            samples_cropped = enc_samples[N_sample*i:N_sample*(i+batch_size)]
            dirs_cropped = enc_dirs[i:i+batch_size].repeat(N_sample,1)
        

        samples_cropped = samples_cropped.to(pose)
        dirs_cropped = dirs_cropped.to(pose)

        output_list.append(nerf(samples_cropped, dirs_cropped))
    
    output = torch.cat(output_list,dim=0)
    output = output.reshape(img_shape[0], img_shape[1],N_sample, 4)
    depth_values = depth_values.to(output)
    rgb, depth_img, acc_map, weights = render(output, depth_values)

    return rgb, depth_img

def nerf_step_sampled(
    nerf_coarse: nn.Module,
    nerf_fine: nn.Module,
    img_shape: tuple,
    sampled_rays_o,
    sampled_rays_d,
    threshold: tuple,
    device,
    pos = True,
    combine = True,
    N_pos = 10,
    N_dir = 4,
    N_sample:int = 64,
    N_importance:int= 128,
    batch_size=32
    ):

    '''
    COARSE MODEL
    '''
    # Sample points along each ray
    samples, depth_values = qurey(sampled_rays_o, sampled_rays_d, threshold, N_sample)

    # flatten the samples
    samples = samples.reshape(-1,3)

    # encode points with positions
    if pos:
        enc_samples = encode_pos(samples, N_pos)
        enc_dirs = encode_pos(sampled_rays_d.reshape(-1,3), N_dir)
    
    # train each pixel
    n = len(enc_dirs)
    output_list = []
    for i in range(0,n,batch_size):
        if (i + batch_size) > n:
            samples_cropped = enc_samples[N_sample*i:]
            dirs_cropped = enc_dirs[i:].repeat(N_sample,1)
        else:
            samples_cropped = enc_samples[N_sample*i:N_sample*(i+batch_size)]
            dirs_cropped = enc_dirs[i:i+batch_size].repeat(N_sample,1)
        

        samples_cropped = samples_cropped.to(device)
        dirs_cropped = dirs_cropped.to(device)

        output_list.append(nerf_coarse(samples_cropped, dirs_cropped))
    
    output = torch.cat(output_list,dim=0)
    output = output.reshape(img_shape[0], img_shape[1],N_sample, 4)
    depth_values = depth_values.to(output)
    rgb_map_coarse, depth_map_coarse, acc_map_coarse, weights = render(output, depth_values)

    # if no fine model just return
    if not nerf_fine:
        return (rgb_map_coarse, depth_map_coarse, acc_map_coarse), (rgb_map_coarse, depth_map_coarse, acc_map_coarse)

    '''
    FINE MODEL
    '''
    if combine:
        samples, depth_values_combined, _, _ = sample_hierarchical(
            sampled_rays_o,
            sampled_rays_d,
            depth_values,
            weights,
            N_importance)
    
        # flatten the samples
        samples = samples.reshape(-1,3)

        # encode points with positions
        if pos:
            enc_samples = encode_pos(samples, N_pos)
            enc_dirs = encode_pos(sampled_rays_d.reshape(-1,3), N_dir)
        
        # train each pixel
        n = len(enc_dirs)
        output_list = []
        for i in range(0,n,batch_size):
            if (i + batch_size) > n:
                samples_cropped = enc_samples[(N_sample+N_importance)*i:]
                dirs_cropped = enc_dirs[i:].repeat((N_sample+N_importance),1)
            else:
                samples_cropped = enc_samples[(N_sample+N_importance)*i:(N_sample+N_importance)*(i+batch_size)]
                dirs_cropped = enc_dirs[i:i+batch_size].repeat((N_sample+N_importance),1)
            

            samples_cropped = samples_cropped.to(device)
            dirs_cropped = dirs_cropped.to(device)

            output_list.append(nerf_fine(samples_cropped, dirs_cropped))
        
        output = torch.cat(output_list,dim=0)
        output = output.reshape(img_shape[0], img_shape[1],(N_sample+N_importance), 4)
        depth_values_combined = depth_values_combined.to(output)
        rgb_map_fine, depth_map_fine, acc_map_fine, weights = render(output, depth_values_combined)

    else:
        _, _, samples, depth_values_importance = sample_hierarchical(
            sampled_rays_o,
            sampled_rays_d,
            depth_values,
            weights,
            N_importance)
        
        # flatten the samples
        samples = samples.reshape(-1,3)

        # encode points with positions
        if pos:
            enc_samples = encode_pos(samples, N_pos)
            enc_dirs = encode_pos(sampled_rays_d.reshape(-1,3), N_dir)
        
        # train each pixel
        n = len(enc_dirs)
        output_list = []
        for i in range(0,n,batch_size):
            if (i + batch_size) > n:
                samples_cropped = enc_samples[N_importance*i:]
                dirs_cropped = enc_dirs[i:].repeat(N_importance,1)
            else:
                samples_cropped = enc_samples[N_importance*i:N_importance*(i+batch_size)]
                dirs_cropped = enc_dirs[i:i+batch_size].repeat(N_importance,1)
            

            samples_cropped = samples_cropped.to(device)
            dirs_cropped = dirs_cropped.to(device)

            output_list.append(nerf_fine(samples_cropped, dirs_cropped))
        
        output = torch.cat(output_list,dim=0)
        output = output.reshape(img_shape[0], img_shape[1],N_importance, 4)
        depth_values_importance = depth_values_importance.to(output)
        rgb_map_fine, depth_map_fine, acc_map_fine, weights = render(output, depth_values_importance)       


    return (rgb_map_coarse, depth_map_coarse, acc_map_coarse), (rgb_map_fine, depth_map_fine, acc_map_fine)