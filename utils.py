import torch 

def roll3D(x, shift):
    """
    Shifts the elements of a 3D tensor along each dimension by the values specified in `shift`.
    """
    return torch.roll(x, shifts=shift, dims=(2, 3, 4))

def pad3D(x, pad=(1, 1, 1, 1, 1, 1)):
    """
    Pads a 3D tensor along each dimension by the values specified in `pad`.
    """
    return F.pad(x, pad)

def pad2D(x, pad=(1, 1, 1, 1)):
    """
    Pads a 2D tensor along each dimension by the values specified in `pad`.
    """
    return F.pad(x, pad)

def Crop3D(x, size):
    """
    Crops a 3D tensor to the specified `size`.
    """
    Z, H, W = size
    return x[:, :Z, :H, :W, :]

def Crop2D(x, size):
    """
    Crops a 2D tensor to the specified `size`.
    """
    H, W = size
    return x[:, :H, :W]

def gen_mask(x):
    """
    Generates a mask for a 3D tensor where two pixels are considered adjacent if they are next to each other in any dimension.
    Non-adjacent elements are marked with -1000.
    """
    mask = torch.zeros_like(x)
    mask[:-1, :, :] = mask[:-1, :, :] + (x[1:, :, :] != x[:-1, :, :]).float() * -1000
    mask[1:, :, :] = mask[1:, :, :] + (x[:-1, :, :] != x[1:, :, :]).float() * -1000
    mask[:, :-1, :] = mask[:, :-1, :] + (x[:, 1:, :] != x[:, :-1, :]).float() * -1000
    mask[:, 1:, :] = mask[:, 1:, :] + (x[:, :-1, :] != x[:, 1:, :]).float() * -1000
    mask[:, :, :-1] = mask[:, :, :-1] + (x[:, :, 1:] != x[:, :, :-1]).float() * -1000
    mask[:, :, 1:] = mask[:, :, 1:] + (x[:, :, :-1] != x[:, :, 1:]).float() * -1000
    return mask

def ConstructTensor(shape, device='cpu', dtype=torch.float32):
    """
    Creates a new tensor with the specified shape.
    """
    return torch.empty(shape, device=device, dtype=dtype)

def TruncatedNormalInit(tensor, mean=0., std=0.02):
    """
    Initializes a tensor with values from a truncated normal distribution.
    """
    with torch.no_grad():
        tensor.fill_(torch.fmod(torch.randn(tensor.shape), 2) * std + mean)

def RangeTensor(start, end, device='cpu', dtype=torch.float32):
    """
    Creates a new tensor with values in the range [start, end).
    """
    return torch.arange(start, end, device=device, dtype=dtype)