import gc
import os

import imageio
import torch
import torch.nn as nn
from torch.optim import Adam
import yaml
from PIL import Image
import numpy as np
from datetime import datetime

from matplotlib import pyplot as plt

from generate_2D_gaussian_splatting import generate_2D_gaussian_splatting
from give_required_data import give_required_data
from ssim import combined_loss


class GaussianSplatting2dOptimizer:

    def __init__(self, config_file_path):
        with open(config_file_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        # Extract values from the loaded config
        self.KERNEL_SIZE = config["KERNEL_SIZE"]
        self.image_size = tuple(config["image_size"])
        self.primary_samples = config["primary_samples"]
        self.backup_samples = config["backup_samples"]
        self.num_epochs = config["num_epochs"]
        self.densification_interval = config["densification_interval"]
        self.learning_rate = config["learning_rate"]
        self.image_file_name = config["image_file_name"]
        self.display_interval = config["display_interval"]
        self.grad_threshold = config["gradient_threshold"]
        self.gauss_threshold = config["gaussian_threshold"]
        self.display_loss = config["display_loss"]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_samples = self.primary_samples + self.backup_samples
        self.starting_size = self.primary_samples
        self.left_over_size = self.backup_samples
        self.persistent_mask = torch.cat(
            [torch.ones(self.starting_size, dtype=bool), torch.zeros(self.left_over_size, dtype=bool)], dim=0)
        self.current_marker = self.starting_size
        self.W_values = None
        self.target_tensor = None
        self.directory = None

    def load_image(self):
        original_image = Image.open(self.image_file_name)
        original_image = original_image.resize((self.image_size[0], self.image_size[0]))
        original_image = original_image.convert('RGB')
        original_array = np.array(original_image)
        original_array = original_array / 255.0
        width, height, _ = original_array.shape

        self.target_tensor = torch.tensor(original_array, dtype=torch.float32, device=self.device)

        coords = np.random.randint(0, [width, height], size=(self.num_samples, 2))
        colour_values, pixel_coords = give_required_data(coords, self.image_size, original_array, self.device)

        pixel_coords = torch.atanh(pixel_coords)

        sigma_values = torch.rand(self.num_samples, 2, device=self.device)
        rho_values = 2 * torch.rand(self.num_samples, 1, device=self.device) - 1
        alpha_values = torch.ones(self.num_samples, 1, device=self.device)
        self.W_values = torch.cat([sigma_values, rho_values, alpha_values, colour_values, pixel_coords], dim=1)

    def create_dir(self):
        # Get current date and time as string
        now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        # Create a directory with the current date and time as its name
        self.directory = f"{now}"
        os.makedirs(self.directory, exist_ok=True)

    def run_opt(self):
        W = nn.Parameter(self.W_values)
        optimizer = Adam([W], lr=self.learning_rate)
        loss_history = []

        for epoch in range(self.num_epochs):
            # find indices to remove and update the persistent mask
            if epoch % (self.densification_interval + 1) == 0 and epoch > 0:
                indices_to_remove = (torch.sigmoid(W[:, 3]) < 0.01).nonzero(as_tuple=True)[0]
                if len(indices_to_remove) > 0:
                    print(f"number of pruned points: {len(indices_to_remove)}")
                self.persistent_mask[indices_to_remove] = False

                # Zero-out parameters and their gradients at every epoch using the persistent mask
                W.data[~self.persistent_mask] = 0.0

            gc.collect()
            torch.cuda.empty_cache()

            output = W[self.persistent_mask]

            batch_size = output.shape[0]

            sigma_x = torch.sigmoid(output[:, 0])
            sigma_y = torch.sigmoid(output[:, 1])
            rho = torch.tanh(output[:, 2])
            alpha = torch.sigmoid(output[:, 3])
            coords = torch.sigmoid(output[:, 4:7])
            pixel_coords = torch.tanh(output[:, 7:9])

            colours_with_alpha = coords * alpha.view(batch_size, 1)
            g_tensor_batch = generate_2D_gaussian_splatting(self.KERNEL_SIZE, sigma_x, sigma_y, rho, pixel_coords,
                                                            colours_with_alpha, self.image_size, self.device)
            loss = combined_loss(g_tensor_batch, self.target_tensor, lambda_param=0.6)

            optimizer.zero_grad()

            loss.backward()

            # Apply zeroing out of gradients at every epoch
            if self.persistent_mask is not None:
                W.grad.data[~self.persistent_mask] = 0.0

            if epoch % self.densification_interval == 0 and epoch > 0:

                # Calculate the norm of gradients
                gradient_norms = torch.norm(W.grad[self.persistent_mask][:, 7:9], dim=1, p=2)
                gaussian_norms = torch.norm(torch.sigmoid(W.data[self.persistent_mask][:, 0:2]), dim=1, p=2)

                sorted_grads, sorted_grads_indices = torch.sort(gradient_norms, descending=True)
                sorted_gauss, sorted_gauss_indices = torch.sort(gaussian_norms, descending=True)

                large_gradient_mask = (sorted_grads > self.grad_threshold)
                large_gradient_indices = sorted_grads_indices[large_gradient_mask]

                large_gauss_mask = (sorted_gauss > self.gauss_threshold)
                large_gauss_indices = sorted_gauss_indices[large_gauss_mask]

                common_indices_mask = torch.isin(large_gradient_indices, large_gauss_indices)
                common_indices = large_gradient_indices[common_indices_mask]
                distinct_indices = large_gradient_indices[~common_indices_mask]

                # Split points with large coordinate gradient and large gaussian values and descale their gaussian
                if len(common_indices) > 0:
                    print(f"number of splitted points: {len(common_indices)}")
                    start_index = self.current_marker + 1
                    end_index = self.current_marker + 1 + len(common_indices)
                    self.persistent_mask[start_index: end_index] = True
                    W.data[start_index:end_index, :] = W.data[common_indices, :]
                    scale_reduction_factor = 1.6
                    W.data[start_index:end_index, 0:2] /= scale_reduction_factor
                    W.data[common_indices, 0:2] /= scale_reduction_factor
                    self.current_marker = self.current_marker + len(common_indices)

                # Clone it points with large coordinate gradient and small gaussian values
                if len(distinct_indices) > 0:
                    print(f"number of cloned points: {len(distinct_indices)}")
                    start_index = self.current_marker + 1
                    end_index = self.current_marker + 1 + len(distinct_indices)
                    self.persistent_mask[start_index: end_index] = True
                    W.data[start_index:end_index, :] = W.data[distinct_indices, :]
                    self.current_marker = self.current_marker + len(distinct_indices)

            optimizer.step()

            loss_history.append(loss.item())

            if epoch % self.display_interval == 0:
                num_subplots = 3 if self.display_loss else 2
                fig_size_width = 18 if self.display_loss else 12

                fig, ax = plt.subplots(1, num_subplots, figsize=(fig_size_width, 6))  # Adjust subplot to 1x3

                generated_array = g_tensor_batch.cpu().detach().numpy()

                ax[0].imshow(g_tensor_batch.cpu().detach().numpy())
                ax[0].set_title('2D Gaussian Splatting')
                ax[0].axis('off')

                ax[1].imshow(self.target_tensor.cpu().detach().numpy())
                ax[1].set_title('Ground Truth')
                ax[1].axis('off')

                if self.display_loss:
                    ax[2].plot(range(epoch + 1), loss_history[:epoch + 1])
                    ax[2].set_title('Loss vs. Epochs')
                    ax[2].set_xlabel('Epoch')
                    ax[2].set_ylabel('Loss')
                    ax[2].set_xlim(0, self.num_epochs)  # Set x-axis limits

                # Display the image
                # plt.show(block=False)
                plt.subplots_adjust(wspace=0.1)  # Adjust this value to your preference
                plt.pause(0.1)  # Brief pause

                img = Image.fromarray((generated_array * 255).astype(np.uint8))

                # Create filename
                filename = f"{epoch}.jpg"

                # Construct the full file path
                file_path = os.path.join(self.directory, filename)

                # Save the image
                img.save(file_path)

                fig.savefig(file_path, bbox_inches='tight')

                plt.clf()  # Clear the current figure
                plt.close()  # Close the current figure

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item()}, on {len(output)} points")

    def save_result(self):
        image_files = []
        for i in range(0, self.num_epochs, self.display_interval):
            image_files.append(f"{i}.jpg")

        # Create a video writer object
        writer = imageio.get_writer(os.path.join(self.directory, 'video.mp4'), fps=2)

        # Add images to the video writer
        for image_file in image_files:
            image_path = os.path.join(self.directory, image_file)
            image = imageio.imread(image_path)
            writer.append_data(image)

        writer.close()
