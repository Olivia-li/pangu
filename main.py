import torch
import torch.nn as nn
import torch.optim as optim
import onnx
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from utils import roll3D, pad3D, pad2D, Crop3D, Crop2D, gen_mask, ConstructTensor, TruncatedNormalInit, RangeTensor
from data import LoadData, LoadConstantMask
from onnx2pytorch import ConvertModel
from timm.models.layers import DropPath
from perlin_numpy import generate_fractal_noise_3d
import torch.onnx
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Inference(input_path, input_surface_path, forecast_range):
    model24 = PanguModel('models/model_1.onnx')
    model24 = model24.to(device)
    # Similarly load other models

    # Load input data
    input_24 = torch.from_numpy(np.load(input_path)).to(device)
    input_surface_24 = torch.from_numpy(np.load(input_surface_path)).to(device)

    input_6, input_surface_6 = input_24, input_surface_24
    input_3, input_surface_3 = input_24, input_surface_24

    output_list = []

    for i in range(forecast_range):
        if (i+1) % 24 == 0:
            input, input_surface = input_24, input_surface_24
            output, output_surface = model24(input, input_surface)
            # Similarly for other models

            output_list.append((output, output_surface))
    return output_list

# Don't need to train the model. This is just an example of how to train the model

# def Train():
#     model = PanguModel()
#     model = model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=3e-6)
#     criterion = nn.L1Loss()

#     epochs = 100
#     for epoch in range(epochs):
#         for step in range(dataset_length):
#             input, input_surface, target, target_surface = LoadData(step)
#             input, input_surface, target, target_surface = input.to(device), input_surface.to(device), target.to(device), target_surface.to(device)

#             optimizer.zero_grad()
#             output, output_surface = model(input, input_surface)
#             loss = criterion(output, target) + 0.25 * criterion(output_surface, target_surface)
#             loss.backward()
#             optimizer.step()

#     torch.save(model.state_dict(), 'models/model.pth')

class PanguModel(nn.Module):
    def __init__(self, onnx_file_path=None):
        super(PanguModel, self).__init__()
        drop_path_list = torch.linspace(0, 0.2, 8)

        # Patch embedding layer
        self._input_layer = PatchEmbedding((2, 4, 4), 192)

        # Four basic layers
        self.layer1 = EarthSpecificLayer(2, 192, drop_path_list[:2], 6)
        self.layer2 = EarthSpecificLayer(6, 384, drop_path_list[6:], 12)
        self.layer3 = EarthSpecificLayer(6, 384, drop_path_list[6:], 12)
        self.layer4 = EarthSpecificLayer(2, 192, drop_path_list[:2], 6)

        # Upsample and downsample layers
        self.upsample = UpSample(384, 192)
        self.downsample = DownSample(192)

        # Patch Recovery layer/
        self._output_layer = PatchRecovery(384, (2, 4, 4))

        # NOTE: Idk what I'm doing lol this shouldn't be here.
        # if onnx_file_path is not None:
        #     onnx_model = onnx.load(onnx_file_path)
        #     pytorch_model = ConvertModel(onnx_model)

        #     # Load the weights from the pytorch model
        #     self.load_state_dict(pytorch_model.state_dict())
        #     # Freeze the model
        #     for param in self.parameters():
        #         param.requires_grad = False


    def forward(self, input, input_surface):
        #Backbone

        # Embed the input fields into patches
        x = self._input_layer(input, input_surface)

        # Encoder, composed of two layers
        # Layer 1, shape (8, 360, 181, C), C = 192 as in the original paper
        x = self.layer1(x, 8, 360, 181) 

        # Store the tensor for skip-connection
        skip = x
        x = self.downsample(x, 8, 360, 181)
        x = self.layer2(x, 8, 180, 91) 
        x = self.layer3(x, 8, 180, 91) 
        x = self.upsample(x)
        x = self.layer4(x, 8, 360, 181) 
        x = torch.cat((skip, x), dim=-1)
        output, output_surface = self._output_layer(x)
        return output, output_surface

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, dim):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv3d(in_channels=5, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        self.conv_surface = nn.Conv2d(in_channels=7, out_channels=dim, kernel_size=patch_size[1:], stride=patch_size[1:])

        # Load constant masks from the disc
        # self.land_mask, self.soil_type, self.topography = LoadConstantMask()
        
    def forward(self, input, input_surface):
        # Zero-pad the input
        input = F.pad(input, (0, 0, 0, 0, 0, 0))  # Assuming you want to pad with 0s on all sides
        input_surface = F.pad(input_surface, (0, 0, 0, 0))  # Assuming you want to pad with 0s on all sides

        # Apply a linear projection for patch_size[0]*patch_size[1]*patch_size[2] patches, patch_size = (2, 4, 4) as in the original paper
        input = self.conv(input)

        # Add three constant fields to the surface fields
        input_surface = torch.cat((input_surface, self.land_mask, self.soil_type, self.topography), dim=1)

        # Apply a linear projection for patch_size[1]*patch_size[2] patches
        input_surface = self.conv_surface(input_surface)

        # Concatenate the input in the pressure level, i.e., in Z dimension
        x = torch.cat((input, input_surface), dim=1)

        # Reshape x for calculation of linear projections
        x = x.permute(0, 2, 3, 4, 1)
        x = x.view(x.shape[0], 8*360*181, x.shape[-1])
        return x

class PatchRecovery(nn.Module):
    def __init__(self, dim, patch_size):
        super(PatchRecovery, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels=dim, out_channels=5, kernel_size=patch_size, stride=patch_size)
        self.conv_surface = nn.ConvTranspose2d(in_channels=dim, out_channels=4, kernel_size=patch_size[1:], stride=patch_size[1:])
        
    def forward(self, x, Z, H, W):
        # The inverse operation of the patch embedding operation, patch_size = (2, 4, 4) as in the original paper
        # Reshape x back to three dimensions
        x = x.permute(0, 2, 1)
        x = x.view(x.shape[0], x.shape[1], Z, H, W)

        # Call the transposed convolution
        output = self.conv(x[:, :, 1:, :, :])
        output_surface = self.conv_surface(x[:, :, 0, :, :])

        # Crop the output to remove zero-paddings
        # Assuming Crop3D and Crop2D are functions you've defined elsewhere to crop the tensor
        output = Crop3D(output)
        output_surface = Crop2D(output_surface)
        return output, output_surface

class DownSample(nn.Module):
    def __init__(self, dim):
        super(DownSample, self).__init__()
        self.linear = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = nn.LayerNorm(4*dim)
  
    def forward(self, x, Z, H, W):
        x = x.view(x.shape[0], Z, H, W, x.shape[-1])
        x = F.pad(x, (0, 0, 0, 0, 0, 0))  # Assuming you want to pad with 0s on all sides
        Z, H, W = x.shape[1:4]
        x = x.view(x.shape[0], Z, H//2, 2, W//2, 2, x.shape[-1])
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        x = x.contiguous().view(x.shape[0], Z*(H//2)*(W//2), 4 * x.shape[-1])
        x = self.norm(x)
        x = self.linear(x)
        return x

class UpSample(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UpSample, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim*4, bias=False)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)
  
    def forward(self, x):
        x = self.linear1(x)
        x = x.view(x.shape[0], 8, 180, 91, 2, 2, x.shape[-1]//4)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        x = x.contiguous().view(x.shape[0], 8, 360, 182, x.shape[-1])
        x = Crop3D(x, (8, 360, 182))  # Assuming Crop3D is a function you've defined elsewhere to crop the tensor
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[-1])
        x = self.norm(x)
        x = self.linear2(x)
        return x

class EarthSpecificLayer(nn.Module):
    def __init__(self, depth, dim, drop_path_ratio_list, heads):
        super(EarthSpecificLayer, self).__init__()
        self.depth = depth
        self.blocks = nn.ModuleList()

        # Construct basic blocks
        for i in range(depth):
            self.blocks.append(EarthSpecificBlock(dim, drop_path_ratio_list, heads))
          
    def forward(self, x, Z, H, W):
        for i in range(self.depth):
            # Roll the input every two blocks
            if i % 2 == 0:
                x = self.blocks[i](x, Z, H, W, roll=False)
            else:
                x = self.blocks[i](x, Z, H, W, roll=True)
        return x

class EarthSpecificBlock(nn.Module):
    def __init__(self, dim, drop_path_ratio, heads):
        super(EarthSpecificBlock, self).__init__()
        self.window_size = (2, 6, 12)
        self.drop_path = DropPath(drop_prob=drop_path_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear = MLP(dim, 0)  # Assuming MLP is a custom PyTorch module you've defined elsewhere
        self.attention = EarthAttention3D(dim, heads, 0, self.window_size)  # Assuming EarthAttention3D is a custom PyTorch module you've defined elsewhere

    def forward(self, x, Z, H, W, roll):
        shortcut = x
        x = x.view(x.shape[0], Z, H, W, x.shape[2])
        x = F.pad(x, (1, 1, 1, 1, 1, 1))  # Assuming you want to pad with 1s on all sides
        ori_shape = x.shape

        if roll:
            x = roll3D(x, shift=[self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2])  # Assuming roll3D is a custom function you've defined elsewhere
            mask = gen_mask(x) #TODO: custom defined function. Unsure if this is correct 
        else:
            mask = torch.zeros_like(x)

        x_window = x.view(x.shape[0], Z//self.window_size[0], self.window_size[0], H // self.window_size[1], self.window_size[1], W // self.window_size[2], self.window_size[2], x.shape[-1])
        x_window = x_window.permute(0, 1, 3, 5, 2, 4, 6, 7)
        x_window = x_window.contiguous().view(-1, self.window_size[0]*self.window_size[1]*self.window_size[2], x.shape[-1])

        x_window = self.attention(x, mask)

        x = x_window.view(-1, Z // self.window_size[0], H // self.window_size[1], W // self.window_size[2], self.window_size[0], self.window_size[1], self.window_size[2], x_window.shape[-1])
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7)
        x = x.contiguous().view(*ori_shape)

        if roll:
            x = roll3D(x, shift=[-self.window_size[0]//2, -self.window_size[1]//2, -self.window_size[2]//2])

        x = Crop3D(x)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[4])

        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.linear(x)))
        return x

class EarthAttention3D(nn.Module):
    def __init__(self, dim, heads, dropout_rate, window_size):
        super(EarthAttention3D, self).__init__()
        self.linear1 = nn.Linear(dim, dim*3, bias=True)
        self.linear2 = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)

        self.head_number = heads
        self.dim = dim
        self.scale = (dim//heads)**-0.5
        self.window_size = window_size

        # For each type of window, we will construct a set of parameters according to the paper
        self.earth_specific_bias = nn.Parameter(torch.empty((2 * window_size[2] - 1) * window_size[1] * window_size[1] * window_size[0] * window_size[0], heads))
        nn.init.trunc_normal_(self.earth_specific_bias, std=0.02)

    #NOTE: removed construct_indux function as it is not used in the forward pass. Potentially need to modify because the input_shape could be different than x.shape in the forward pass
    def forward(self, x, mask):
        # Linear layer to create query, key and value
        x = self.linear1(x)

        # Record the original shape of the input
        original_shape = x.shape

        # reshape the data to calculate multi-head attention
        qkv = x.view(*x.shape[:-1], 3, self.head_number, self.dim // self.head_number)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.split(1, dim=0)

        # Scale the attention
        query = query * self.scale

        # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
        attention = torch.matmul(query, key.transpose(-2, -1))

        # Add the Earth-Specific bias to the attention matrix
        attention = attention + self.earth_specific_bias

        # Mask the attention between non-adjacent pixels, e.g., simply add -100 to the masked element.
        attention = attention.masked_fill(mask == 0, -1e9)
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        # Calculated the tensor after spatial mixing.
        x = torch.matmul(attention, value.transpose(-2, -1))

        # Reshape tensor to the original shape
        x = x.permute(1, 2, 0, 3).contiguous()
        x = x.view(*original_shape)

        # Linear layer to post-process operated tensor
        x = self.linear2(x)
        x = self.dropout(x)
        return x
  
class MLP(nn.Module):
    def __init__(self, dim, dropout_rate):
        '''MLP layers, same as most vision transformer architectures.'''
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x


def PerlinNoise():
    octaves = 3
    noise_scale = 0.2
    period_number = 12
    H, W = 721, 1440
    persistence = 0.5
    perlin_noise = noise_scale*generate_fractal_noise_3d((H, W), (period_number, period_number), octaves, persistence)
    return perlin_noise

def main():
    # Load the onnx data into PanguModel
    model = PanguModel('models/model_1.onnx')


if __name__ == "__main__":
    main()