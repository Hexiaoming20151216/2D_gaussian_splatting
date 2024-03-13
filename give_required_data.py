import torch
import numpy as np


def give_required_data(input_coords, image_size_in, image_array, device):
    # normalising pixel coordinates [-1,1]
    coords_in = torch.tensor(input_coords / [image_size_in[0], image_size_in[1]], device=device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=device).float()
    coords_in = (center_coords_normalized - coords_in) * 2.0

    # Fetching the colour of the pixels in each coordinates
    colour_values_in = [image_array[coord[1], coord[0]] for coord in input_coords]
    colour_values_np = np.array(colour_values_in)
    colour_values_tensor = torch.tensor(colour_values_np, device=device).float()

    return colour_values_tensor, coords_in
