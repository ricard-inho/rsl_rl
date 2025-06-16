# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,  # Consider this as the flat size if no CNN, or a dummy if CNN is used
        num_critic_obs, # Consider this as the flat size if no CNN, or a dummy if CNN is used
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        clip_actions: bool = False,
        clip_actions_range: tuple = (-1.0, 1.0),
        actor_cnn_config: dict = None,
        critic_cnn_config: dict = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # --- Actor Network ---
        self.actor_cnn = None
        mlp_input_dim_a = num_actor_obs # Default to num_actor_obs

        if actor_cnn_config:
            print("Configuring Actor CNN...")
            # Extract image dimensions from config
            image_channels = actor_cnn_config.get("image_channels", 3)
            image_height = actor_cnn_config.get("image_height")
            image_width = actor_cnn_config.get("image_width")

            if image_height is None or image_width is None:
                raise ValueError("actor_cnn_config must specify 'image_height' and 'image_width' for CNNs.")

            cnn_layers_actor = []
            current_channels = image_channels
            current_height = image_height
            current_width = image_width

            for conv_params in actor_cnn_config.get("conv_layers", []):
                cnn_layers_actor.append(nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=conv_params["out_channels"],
                    kernel_size=conv_params["kernel_size"],
                    stride=conv_params["stride"],
                    padding=conv_params.get("padding", 0)
                ))
                cnn_layers_actor.append(activation)
                # Update current dimensions after convolution
                current_height = (current_height + 2 * conv_params.get("padding", 0) - conv_params["kernel_size"]) // conv_params["stride"] + 1
                current_width = (current_width + 2 * conv_params.get("padding", 0) - conv_params["kernel_size"]) // conv_params["stride"] + 1
                current_channels = conv_params["out_channels"]
            
            for pool_params in actor_cnn_config.get("pool_layers", []):
                cnn_layers_actor.append(nn.MaxPool2d(
                    kernel_size=pool_params["kernel_size"],
                    stride=pool_params["stride"],
                    padding=pool_params.get("padding", 0),
                    dilation=pool_params.get("dilation", 1)
                ))
                # Update current dimensions after pooling
                current_height = (current_height + 2 * pool_params.get("padding", 0) - pool_params["kernel_size"]) // pool_params["stride"] + 1
                current_width = (current_width + 2 * pool_params.get("padding", 0) - pool_params["kernel_size"]) // pool_params["stride"] + 1
                
            self.actor_cnn = nn.Sequential(*cnn_layers_actor)

            # Calculate the flattened output dimension of the CNN
            dummy_input_actor = torch.zeros(1, image_channels, image_height, image_width)
            cnn_output_shape_actor = self.actor_cnn(dummy_input_actor).shape
            mlp_input_dim_a = cnn_output_shape_actor[1] * cnn_output_shape_actor[2] * cnn_output_shape_actor[3]
            print(f"Actor CNN output flattened dimension: {mlp_input_dim_a}")
        else:
            print("Actor CNN is not configured. Using MLP directly.")
            
        # Actor MLP
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor_mlp = nn.Sequential(*actor_layers) # Renamed to actor_mlp for clarity

        # --- Critic Network ---
        self.critic_cnn = None
        mlp_input_dim_c = num_critic_obs # Default to num_critic_obs

        if critic_cnn_config:
            print("Configuring Critic CNN...")
            image_channels = critic_cnn_config.get("image_channels", 3)
            image_height = critic_cnn_config.get("image_height")
            image_width = critic_cnn_config.get("image_width")

            if image_height is None or image_width is None:
                raise ValueError("critic_cnn_config must specify 'image_height' and 'image_width' for CNNs.")

            cnn_layers_critic = []
            current_channels = image_channels
            current_height = image_height
            current_width = image_width

            for conv_params in critic_cnn_config.get("conv_layers", []):
                cnn_layers_critic.append(nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=conv_params["out_channels"],
                    kernel_size=conv_params["kernel_size"],
                    stride=conv_params["stride"],
                    padding=conv_params.get("padding", 0)
                ))
                cnn_layers_critic.append(activation)
                current_height = (current_height + 2 * conv_params.get("padding", 0) - conv_params["kernel_size"]) // conv_params["stride"] + 1
                current_width = (current_width + 2 * conv_params.get("padding", 0) - conv_params["kernel_size"]) // conv_params["stride"] + 1
                current_channels = conv_params["out_channels"]
            
            for pool_params in critic_cnn_config.get("pool_layers", []):
                cnn_layers_critic.append(nn.MaxPool2d(
                    kernel_size=pool_params["kernel_size"],
                    stride=pool_params["stride"],
                    padding=pool_params.get("padding", 0),
                    dilation=pool_params.get("dilation", 1)
                ))
                current_height = (current_height + 2 * pool_params.get("padding", 0) - pool_params["kernel_size"]) // pool_params["stride"] + 1
                current_width = (current_width + 2 * pool_params.get("padding", 0) - pool_params["kernel_size"]) // pool_params["stride"] + 1

            self.critic_cnn = nn.Sequential(*cnn_layers_critic)

            # Calculate the flattened output dimension of the CNN
            dummy_input_critic = torch.zeros(1, image_channels, image_height, image_width)
            cnn_output_shape_critic = self.critic_cnn(dummy_input_critic).shape
            mlp_input_dim_c = cnn_output_shape_critic[1] * cnn_output_shape_critic[2] * cnn_output_shape_critic[3]
            print(f"Critic CNN output flattened dimension: {mlp_input_dim_c}")
        else:
            print("Critic CNN is not configured. Using MLP directly.")

        # Value function (Critic MLP)
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic_mlp = nn.Sequential(*critic_layers) # Renamed to critic_mlp

        print(f"Actor MLP: {self.actor_mlp}")
        print(f"Critic MLP: {self.critic_mlp}")

        # Action clipping parameters
        self.clip_actions = clip_actions
        self.clip_actions_range = clip_actions_range
        if self.clip_actions:
            self.clipping_layer = nn.Tanh()
            
        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        mode = self.distribution.mean
        if self.clip_actions:
            # Note: This is an action-space clipping, not the input to the distribution.
            # The mean from the distribution is what's being rescaled here.
            mode = ((mode + 1) /2.0)* (self.clip_actions_range[1] - self.clip_actions_range[0]) + self.clip_actions_range[0]
        return mode
        
    @property
    def action_std(self):
        return self.distribution.stddev
        
    @property
    def actions_distribution(self) -> torch.Tensor:
        # Mean and Std concatenated on an extra dimension
        return torch.stack([self.distribution.mean, self.distribution.stddev], dim=-1)

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def build_distribution(self, parameters):
        # build the distribution
        return Normal(parameters[..., 0], parameters[..., 1])

    def update_distribution(self, observations):
        # Process observations through CNN if configured
        if self.actor_cnn:
            # Assuming observations are already in (batch_size, channels, height, width)
            x = self.actor_cnn(observations)
            x = x.view(x.size(0), -1) # Flatten the CNN output
        else:
            x = observations # Use observations directly if no CNN

        # compute mean
        mean = self.actor_mlp(x) # Use actor_mlp
        if self.clip_actions:
            mean = self.clipping_layer(mean) # Apply tanh to the raw action mean from the MLP

        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        act = self.distribution.sample()
        if self.clip_actions:
            # Apply tanh to clip the sampled actions to [-1, 1]
            act = self.clipping_layer(act)
            # Rescale the actions to the desired range
            act = ((act + 1) / 2.0) * (self.clip_actions_range[1] - self.clip_actions_range[0]) + self.clip_actions_range[0]
        return act

    def get_actions_log_prob(self, actions):
        if self.clip_actions:
            # The actions passed here are assumed to be in the desired output range [clip_actions_range[0], clip_actions_range[1]]
            # We need to reverse the scaling and tanh to get the value that would be sampled from the Gaussian.
            
            # 1. Reverse the scaling to get actions in [-1, 1] range (unscaled_actions)
            unscaled_actions = (actions - self.clip_actions_range[0]) / (self.clip_actions_range[1] - self.clip_actions_range[0]) * 2.0 - 1.0
            
            # 2. Revert the tanh to get the original actions (gaussian_actions) that would be sampled from the Normal distribution.
            # Use the static method for numerical stability.
            gaussian_actions = ActorCritic.inverse_tanh(unscaled_actions)
            
            # Compute log probability with the correction for tanh transformation
            # The log_prob of a transformed distribution is log_prob(original) - log(|jacobian_determinant|)
            # For tanh, the derivative is 1 - tanh^2(x).
            # log(|d(tanh(x))/dx|) = log(1 - tanh^2(x))
            # Since unscaled_actions = tanh(gaussian_actions), we use unscaled_actions^2
            log_prob = self.distribution.log_prob(gaussian_actions).sum(dim=-1)
            log_det_jacobian = torch.log(1 - unscaled_actions.pow(2) + 1e-6).sum(dim=-1) # Add epsilon for stability
            return log_prob - log_det_jacobian
        else:
            return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        # Process observations through CNN if configured
        if self.actor_cnn:
            x = self.actor_cnn(observations)
            x = x.view(x.size(0), -1) # Flatten the CNN output
        else:
            x = observations # Use observations directly if no CNN

        mode = self.actor_mlp(x) # Use actor_mlp
        if self.clip_actions:
            # Apply tanh to the raw action mean from the MLP
            mode = self.clipping_layer(mode)
            # Rescale the actions to the desired range
            mode = ((mode + 1) / 2.0) * (self.clip_actions_range[1] - self.clip_actions_range[0]) + self.clip_actions_range[0]
        return mode

    def evaluate(self, critic_observations, **kwargs):
        # Process critic_observations through CNN if configured
        if self.critic_cnn:
            x = self.critic_cnn(critic_observations)
            x = x.view(x.size(0), -1) # Flatten the CNN output
        else:
            x = critic_observations # Use observations directly if no CNN

        value = self.critic_mlp(x) # Use critic_mlp
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
        
    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())
        
    @staticmethod
    def inverse_tanh(y):
        # Clamp to avoid numerical issues when y is exactly +/-1.0
        eps = torch.finfo(y.dtype).eps
        return ActorCritic.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))