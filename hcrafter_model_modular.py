# hcrafter_model_modular.py

import torch
from torch import nn, Tensor

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.model.model_utils import nonlinearity, create_mlp
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.utils import log


# STEP 1: An exact copy of the vision encoder from your non-modular implementation.
# This ensures both models use the same architecture for processing pixels,
# creating a fair and controlled experiment.
class HcrafterConvEncoder(Encoder):
    """
    This is the vision processing backbone, copied from the non-modular file.
    It expects a dictionary observation and processes the 'obs' key.
    """
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        if not hasattr(obs_space, 'keys') or 'obs' not in obs_space.keys():
            raise TypeError(f"HcrafterConvEncoder expects a dict obs_space with key 'obs', but got {type(obs_space)}")

        self.basic_encoder = nn.Sequential(
            nn.Conv2d(obs_space['obs'].shape[0], 32, 8, 4),
            nonlinearity(cfg),
            nn.Conv2d(32, 64, 4, 2),
            nonlinearity(cfg),
            nn.Conv2d(64, 64, 3, 1),
            nonlinearity(cfg),
            nn.Flatten(),
        )
        self.encoder_out_size = calc_num_elements(self.basic_encoder, obs_space['obs'].shape)
        log.debug(f"Copied Vision Encoder output size: {self.encoder_out_size}")

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def forward(self, obs_dict: dict) -> Tensor:
        return self.basic_encoder(obs_dict['obs'])

# ------------------------------------------------------------------------------

# STEP 2: The main modular encoder that uses the vision encoder as its base.
class ModularHcrafterEncoder(Encoder):
    """
    A modular encoder that uses the HcrafterConvEncoder for vision processing
    and then splits the logic into multiple 'module heads' for higher-level reasoning.
    """
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        # Use the copied HcrafterConvEncoder as the shared "eyeballs"
        self.vision_encoder = HcrafterConvEncoder(cfg, obs_space)
        vision_out_size = self.vision_encoder.get_out_size()

        # Handle measurements flexibly, just like the original non-modular code
        self.num_measurements = 0
        if "measurements" in obs_space.keys():
            self.num_measurements = obs_space["measurements"].shape[0]
        log.debug(f"Modular encoder found {self.num_measurements} measurements.")

        # Determine the number of modules. Fallback to 1 if no measurements exist.
        self.num_modules = self.num_measurements if self.num_measurements > 0 else 1

        # Define module heads that process the combined vision and measurement data
        module_input_size = vision_out_size + self.num_measurements
        self.module_head_out_size = 512

        self.module_heads = nn.ModuleList()
        for i in range(self.num_modules):
            head = create_mlp(
                layer_sizes=[128, self.module_head_out_size],
                input_size=module_input_size,
                activation=nonlinearity(cfg),
            )
            self.module_heads.append(head)

        log.debug(f"Created {self.num_modules} module heads. Final output size: {self.get_out_size()}")

    def get_out_size(self) -> int:
        return self.module_head_out_size

    def forward(self, obs_dict: dict):
        # Get features from the shared vision architecture
        vision_features = self.vision_encoder(obs_dict)

        # Create the shared representation for the module heads
        if self.num_measurements > 0:
            measurements = obs_dict['measurements'].float()
            shared_representation = torch.cat((vision_features, measurements), dim=1)
        else:
            shared_representation = vision_features

        # Pass the representation through each module head
        module_outputs = [head(shared_representation) for head in self.module_heads]

        # Aggregate module outputs by element-wise summation ('gmQ' style)
        aggregated_output = torch.sum(torch.stack(module_outputs), dim=0)

        return aggregated_output

# ------------------------------------------------------------------------------

# STEP 3: The factory function required by sample-factory to build your model.
def make_modular_hcrafter_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function for the modular encoder."""
    return ModularHcrafterEncoder(cfg, obs_space)