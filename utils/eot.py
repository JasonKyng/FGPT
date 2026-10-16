import torch
from einops import rearrange

def OptimalTransport(args, image_features, text_features, entropy=None, prob=None):
    b, M, _ = image_features.shape
    sim_op_images = []
    sim_op_pixels = []
    for text_features_ in text_features:
        b, N, _ = text_features_.shape
        n_cls = 1
        sim = torch.einsum('bmd,bnd->bmn', image_features, text_features_).contiguous() 
        wdist = 1.0 - sim
        if entropy is None:
            xx=torch.zeros(b*n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        else:
            xx=entropy.softmax(-1).view(b*n_cls, M)
        if prob is None:
            yy=torch.zeros(b*n_cls, N, dtype=sim.dtype, device=sim.device).fill_(1. / N)
        else:
            yy=torch.stack(torch.chunk(prob, dim = 1, chunks = 2), dim = 1).softmax(-1).view(b*n_cls, N)

        with torch.no_grad():
            KK = torch.exp(-wdist / args.eps)
            T = Sinkhorn(KK, xx, yy, args.ot_threshold)
        if torch.isnan(T).any():
            return None
        
        sim_op = T*sim
        sim_op_image = torch.sum(sim_op, dim=(1, 2))
        sim_op_image = rearrange(sim_op_image, '(b n_cls) -> b n_cls', b=b, n_cls=n_cls)

        sim_op_pixel = torch.sum(sim_op, dim=-1)
        sim_op_pixel = rearrange(sim_op_pixel, '(b n_cls) M -> b M n_cls', b=b, M=M, n_cls=n_cls)
        sim_op_images.append(sim_op_image)
        sim_op_pixels.append(sim_op_pixel)
    return torch.cat(sim_op_images, dim = -1).softmax(dim=-1), torch.cat(sim_op_pixels, dim = -1)

def Sinkhorn(K, u, v, thresh):
    
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    
    for i in range(50):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

    return T