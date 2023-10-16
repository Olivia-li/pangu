import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
# from Your_Data_Code import LoadData, LoadConstantMask, LoadStatic
from perlin_numpy import generate_fractal_noise_3d
import torch.onnx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        self.upsample = UpSample(384, 192)
        self.downsample = DownSample(192)

        self._output_layer = PatchRecovery(384)

        if onnx_file_path is not None:
            onnx_model = onnx.load(onnx_file_path)
            torch.onnx.import_graph(onnx_model.graph, self)

    def forward(self, input, input_surface):
        x = self._input_layer(input, input_surface)
        x = self.layer1(x, 8, 360, 181) 
        skip = x
        x = self.downsample(x, 8, 360, 181)
        x = self.layer2(x, 8, 180, 91) 
        x = self.layer3(x, 8, 180, 91) 
        x = self.upsample(x)
        x = self.layer4(x, 8, 360, 181) 
        x = torch.cat((skip, x), dim=-1)
        output, output_surface = self._output_layer(x)
        return output, output_surface

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

def Inference(input_path, input_surface_path, forecast_range):
    model24 = PanguModel('models/model_1.pth')
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

def PerlinNoise():
    octaves = 3
    noise_scale = 0.2
    period_number = 12
    H, W = 721, 1440
    persistence = 0.5
    perlin_noise = noise_scale*generate_fractal_noise_3d((H, W), (period_number, period_number), octaves, persistence)
    return perlin_noise

class EarthSpecificLayer(nn.Module):
  def __init__(self, depth, dim, drop_path_ratio_list, heads):
    super(EarthSpecificLayer, self).__init__()
    self.depth = depth
    self.blocks = nn.ModuleList([EarthSpecificBlock(dim, drop_path_ratio_list[i], heads) for i in range(depth)])

  def forward(self, x, Z, H, W):
    for i in range(self.depth):
        if i % 2 == 0:
            x = self.blocks[i](x, Z, H, W, roll=False)
        else:
            x = self.blocks[i](x, Z, H, W, roll=True)
    return x

class EarthSpecificBlock(nn.Module):
    def __init__(self, dim, drop_path_ratio, heads):
        super(EarthSpecificBlock, self).__init__()
        self.window_size = (2, 6, 12)
        self.drop_path = DropPath(drop_path_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear = MLP(dim, 0)
        self.attention = EarthAttention3D(dim, heads, 0, self.window_size)

    def forward(self, x, Z, H, W, roll):
        shortcut = x
        x = reshape(x, (x.shape[0], Z, H, W, x.shape[2]))
        x = pad3D(x)
        ori_shape = x.shape
        if roll:
            x = roll3D(x, shift=[self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2])
            mask = gen_mask(x)
        else:
            mask = no_mask
            x_window = reshape(x, (x.shape[0], Z//window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], x.shape[-1]))
            x_window = TransposeDimensions(x_window, (0, 1, 3, 5, 2, 4, 6, 7))
            x_window = reshape(x_window, (-1, window_size[0]* window_size[1]*window_size[2], x.shape[-1]))
            x_window = self.attention(x, mask)
            x = reshape(x_window, (-1, Z // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], x_window.shape[-1]))
            x = TransposeDimensions(x, (0, 1, 4, 2, 5, 3, 6, 7))
            x = reshape(x_window, ori_shape)
        if roll:
            x = roll3D(x, shift=[-self.window_size[0]//2, -self.window_size[1]//2, -self.window_size[2]//2])
            x = Crop3D(x)
            x = reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[4]))
            x = shortcut + self.drop_path(self.norm1(x))
            x = x + self.drop_path(self.norm2(self.linear(x)))
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
        '''Generate random Perlin noise: we follow https://github.com/pvigier/perlin-numpy/ to calculate the perlin noise.'''
        # Define number of noise
        octaves = 3
        # Define the scaling factor of noise
        noise_scale = 0.2
        # Define the number of periods of noise along the axis
        period_number = 12
        # The size of an input slice
        H, W = 721, 1440
        # Scaling factor between two octaves
        persistence = 0.5
        # see https://github.com/pvigier/perlin-numpy/ for the implementation of GenerateFractalNoise (e.g., from perlin_numpy import generate_fractal_noise_3d)
        perlin_noise = noise_scale*generate_fractal_noise_3d((H, W), (period_number, period_number), octaves, persistence)
        return perlin_noise

def main():
    # Define the paths to your input data and the forecast range
    input_path = 'input_data/input.npy'
    input_surface_path = 'input_data/input_surface.npy'
    forecast_range = 10  # replace with your actual forecast range

    # Call the Inference function
    output_list = Inference(input_path, input_surface_path, forecast_range)

    # Now output_list contains the output of the model for each forecast step
    # You can process the output_list as needed

if __name__ == "__main__":
    main()