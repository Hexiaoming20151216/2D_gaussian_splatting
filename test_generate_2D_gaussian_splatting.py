import torch
from matplotlib import pyplot as plt

from generate_2D_gaussian_splatting import generate_2D_gaussian_splatting

device = "cpu"

kernel_size = 3  # You can adjust the kernel size as needed
sigma_x = torch.tensor([2.0, 0.5, 0.5, 0.2], device=device)
sigma_y = torch.tensor([2.0, 1.5, 1.5, 0.3], device=device)
rho = torch.tensor([0.0, -0.9, 0.9, 0.1], device=device)
coords = torch.tensor([(-0.5, -0.5), (-0.3, 0.8), (0.5, 0.5), (0.1, 0.1)], device=device)
colors = torch.tensor([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)], device=device)
img_size = (256, 256, 3)

final_image = generate_2D_gaussian_splatting(kernel_size, sigma_x, sigma_y, rho, coords, colors, img_size,
                                             device_in=device)

plt.imshow(final_image.detach().cpu().numpy())
plt.axis("off")
plt.tight_layout()
plt.show()
