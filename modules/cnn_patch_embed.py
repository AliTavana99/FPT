import torch
import torch.nn as nn

class CNNPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size = config.image_size if isinstance(config.image_size, (tuple, list)) else (config.image_size, config.image_size)
        patch_size = config.patch_size if isinstance(config.patch_size, (tuple, list)) else (config.patch_size, config.patch_size)
        num_channels = config.num_channels
        hidden_size = config.hidden_size

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # CNN architecture: Multiple conv layers for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_size, kernel_size=patch_size, stride=patch_size),
        )

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}x{width}) doesn't match model ({self.image_size[0]}x{self.image_size[1]})."
                )
        embeddings = self.cnn(pixel_values).flatten(2).transpose(1, 2)
        return embeddings
