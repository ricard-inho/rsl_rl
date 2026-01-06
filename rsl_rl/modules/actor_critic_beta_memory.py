# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Beta

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.memory_network import HybridMemoryActorNetwork, PureMemoryActorNetwork


class ActorCriticBetaMemory(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        clip_actions: bool = True,
        clip_actions_range: tuple = (-1.0, 1.0),
        use_embeddings=True,
        embeddings_size=64,
        generator_size=(256,256,256),
        num_memory_obs=64,
        network_type:str="hybrid",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        # actor_layers = []
        # actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        # actor_layers.append(activation)
        # for layer_index in range(len(actor_hidden_dims)):
        #     if layer_index == len(actor_hidden_dims) - 1:
        #         self.alpha = nn.Linear(actor_hidden_dims[layer_index], num_actions)
        #         self.beta = nn.Linear(actor_hidden_dims[layer_index], num_actions)
        #         self.alpha_activation = nn.Softplus()
        #         self.beta_activation = nn.Softplus()
        #     else:
        #         actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
        #         actor_layers.append(activation)
        # self.actor = nn.Sequential(*actor_layers)

        if network_type == "hybrid":
            self.actor = HybridMemoryActorNetwork(
                num_actor_obs,
                num_memory_obs,
                num_actions,
                actor_hidden_dims=actor_hidden_dims,
                use_embeddings=use_embeddings,
                embeddings_size=embeddings_size,
                generator_size=generator_size,
                activation=activation,
                device="cuda",
                dtype=torch.float32,
            )

            self.critics_list = nn.ModuleList()
            num_tasks = 4  # Assuming 4 tasks; adjust as necessary
            for _ in range(num_tasks):
                task_critic_layers = []
                task_critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
                task_critic_layers.append(activation)
                for layer_index in range(len(critic_hidden_dims)):
                    if layer_index == len(critic_hidden_dims) - 1:
                        task_critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
                    else:
                        task_critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                        task_critic_layers.append(activation)
                task_critic = nn.Sequential(*task_critic_layers)
                self.critics_list.append(task_critic)

        
            
        elif network_type == "pure":
            self.actor = PureMemoryActorNetwork(
                num_actor_obs,
                num_memory_obs,
                num_actions,
                actor_hidden_dims=actor_hidden_dims,
                use_embeddings=use_embeddings,
                embeddings_size=embeddings_size,
                generator_size=generator_size,
                activation=activation,
                device="cuda",
                dtype=torch.float32,
            )
            self.critic = PureMemoryActorNetwork(
                num_critic_obs,
                num_memory_obs, 
                1, # Single value output
                actor_hidden_dims=critic_hidden_dims,
                use_embeddings=use_embeddings,
                embeddings_size=embeddings_size,
                generator_size=generator_size,
                activation=activation,
                device="cuda",
                dtype=torch.float32,
            )

        self.alpha = nn.Linear(num_actions, num_actions)
        self.beta = nn.Linear(num_actions, num_actions)
        self.alpha_activation = nn.Softplus()
        self.beta_activation = nn.Softplus()

        self.clip_actions_range = clip_actions_range

        # Value function
        # critic_layers = []
        # critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        # critic_layers.append(activation)
        # for layer_index in range(len(critic_hidden_dims)):
        #     if layer_index == len(critic_hidden_dims) - 1:
        #         critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
        #     else:
        #         critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
        #         critic_layers.append(activation)
        # self.critic = nn.Sequential(*critic_layers)

        print(f"Actor Beta MLP: {self.actor}")
        print(f"Critic MLP: {self.critics_list}")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        self.a = None
        self.b = None
        # disable args validation for speedup
        Beta.set_default_validate_args(False)

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
        mode = self.a / (self.a + self.b)
        mode_rescaled = mode * (self.clip_actions_range[1] - self.clip_actions_range[0]) + self.clip_actions_range[0]
        return mode_rescaled
    
    @property
    def action_std(self):
        return torch.sqrt(self.a * self.b / ((self.a + self.b + 1) * (self.a + self.b) ** 2))

    @property
    def actions_distribution(self):
        # Alpha and beta concatenated on an extra dimension
        return torch.stack([self.a, self.b], dim=-1)
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def build_distribution(self, parameters):
        # create distribution
        return Beta(parameters[...,0], parameters[...,1])

    def update_distribution(self, observations):
        # compute mean
        latent = self.actor(observations["general_obs"], observations["semantic_emb"])
        self.a = self.alpha_activation(self.alpha(latent)) + 1.0
        self.b = self.beta_activation(self.beta(latent)) + 1.0

        # create distribution
        self.distribution = Beta(self.a, self.b)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        act = self.distribution.sample()
        act_rescaled = act * (self.clip_actions_range[1] - self.clip_actions_range[0]) + self.clip_actions_range[0]
        return act_rescaled

    def get_actions_log_prob(self, actions):
        # Unscale the actions to [0, 1] before computing the log probability.
        unscaled_actions = (actions - self.clip_actions_range[0]) / (self.clip_actions_range[1] - self.clip_actions_range[0])
        # For numerical stability, clip the actions to [1e-5, 1 - 1e-5].
        unscaled_actions = torch.clamp(unscaled_actions, 1e-5, 1 - 1e-5)
        return self.distribution.log_prob(unscaled_actions).sum(dim=-1)

    def act_inference(self, observations):
        latent = self.actor(observations["general_obs"], observations["semantic_emb"])
        self.a = self.alpha_activation(self.alpha(latent))
        self.b = self.beta_activation(self.beta(latent))
        mode = self.a / (self.a + self.b)
        mode_rescaled = mode * (self.clip_actions_range[1] - self.clip_actions_range[0]) + self.clip_actions_range[0]
        return mode_rescaled

    def evaluate(self, critic_observations, **kwargs):
        # Determine the task ID for every sample in the batch
        # task_ids shape: (Batch_size,)
        task_ids = critic_observations['task_id_one_hot'].argmax(dim=1)

        # 2. Initialize the final value tensor
        # value shape: (Batch_size, 1)
        batch_size = critic_observations['general_obs'].shape[0]
        value = torch.zeros(batch_size, 1, 
                            device=critic_observations['general_obs'].device)
        
        # 3. Iterate over the critics/tasks and compute values selectively
        # Use enumerate on the critics list, which gives the task_id (i) and the model (critic)
        # Note: Make sure self.critics_list length matches your num_tasks (e.g., 4)
        for i, critic in enumerate(self.critics_list):
            # Create a mask for samples belonging to the current task i
            mask = (task_ids == i)
            
            # Only compute values if there are samples for this task in the batch
            if mask.any():
                # Get the observations for the current task
                obs_for_task = critic_observations['general_obs'][mask]
                
                # Compute values using the specific critic for task i
                # values shape: (N_samples_for_task, 1)
                values = critic(obs_for_task)
                
                # Scatter the results back into the final value tensor using the same mask
                value[mask] = values

        # value = self.critic(critic_observations)
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
