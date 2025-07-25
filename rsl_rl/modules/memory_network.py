# import torch
# import math
# from rsl_rl.utils import resolve_nn_activation
# from torch.nn.utils.parametrizations import spectral_norm


# class HybridMemoryNetwork:
#     def __init__(self):
#         pass


# class MemoryShaper(torch.nn.Module):
#     def __init__(
#             self,
#             input_size,
#             target_size,
#             device=None,
#             dtype=None,
#     ) -> None:
#         super().__init__()
#         self._K = torch.nn.Linear(input_size, target_size[0], device=device, dtype=dtype)
#         self._Q = torch.nn.Linear(input_size, target_size[1], device=device, dtype=dtype)

#     def forward(self, input_data):
#         key = self._K(input_data).unsqueeze(-1)
#         query = self._Q(input_data).unsqueeze(-1)
#         d_k = query.size(-1)
#         scores = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#         return scores

# class HybridMemoryRetrieverHead(torch.nn.Module):
#     def __init__(
#             self,
#             input_data_size: int = 12,
#             use_embeddings: bool = True,
#             embeddings_size: int = 32,
#             generator_size: tuple = (256,256),
#             target_network_size: tuple = (32,32),
#             generate_bias: bool = True,
#             activation:str = "elu",
#             device=None,
#             dtype=None,
#             **kwargs,
#     ) -> None:
#         super().__init__()
#         if isinstance(activation, str):
#             activation = resolve_nn_activation(activation)

#         self._use_embeddings = use_embeddings
#         if use_embeddings:
#             self.embeddings = torch.nn.Linear(input_data_size, embeddings_size, device=device, dtype=dtype)
#             input_data_size = embeddings_size
        
#         generator = [torch.nn.Linear(input_data_size, generator_size[0], device=device, dtype=dtype)]
#         generator.append(activation)
#         for i in range(len(generator_size) -1):
#             generator.append(torch.nn.Linear(generator_size[i], generator_size[i+1], device=device, dtype=dtype))
#             generator.append(activation)
#         self._generator = torch.nn.Sequential(*generator)
        
#         self._weight_scaling_head = MemoryShaper(generator_size[-1], target_network_size, device=device, dtype=dtype)
#         self._weight_additive_head = MemoryShaper(generator_size[-1], target_network_size, device=device, dtype=dtype)

#         self._generate_bias = generate_bias
#         if generate_bias:
#             bias_size = (target_network_size[-1], 1)
#             self._bias_scaling_head = MemoryShaper(generator_size[-1], bias_size, device=device, dtype=dtype)
#             self._bias_additive_head = MemoryShaper(generator_size[-1], bias_size, device=device, dtype=dtype)
#         else:
#             self._default_bias_scaling = torch.ones((1, generator_size[-1]), device=device, dtype=dtype)
#             self._default_bias_offset = torch.zeros((1, generator_size[-1]), device=device, dtype=dtype)

#     def forward(self, input_data):
#         if self._use_embeddings:
#             input_data = self.embeddings(input_data)

#         generator_output = self._generator(input_data)

#         # Generate weights
#         weight_scaling = self._weight_scaling_head(generator_output)
#         weight_offset = self._weight_additive_head(generator_output)
#         # Generate bias if requested else use default
#         if self._generate_bias:
#             bias_scaling = self._bias_scaling_head(generator_output).squeeze(1)
#             bias_offset = self._bias_additive_head(generator_output).squeeze(1)
#         else:
#             bias_scaling = self._default_bias_scaling.repeat(input_data.shape[0],1)
#             bias_offset = self._default_bias_offset.repeat(input_data.shape[0],1)
#         return weight_scaling, weight_offset, bias_scaling, bias_offset

# class PureMemoryRetrieverHead(torch.nn.Module):
#     def __init__(
#             self,
#             input_data_size: int = 12,
#             use_embeddings: bool = True,
#             embeddings_size: int = 32,
#             generator_size: tuple = (256,256),
#             target_network_size: tuple = (32,32),
#             generate_bias: bool = True,
#             activation:str = "elu",
#             device=None,
#             dtype=None,
#             **kwargs,
#     ) -> None:
#         super().__init__()
#         if isinstance(activation, str):
#             activation = resolve_nn_activation(activation)

#         self._use_embeddings = use_embeddings
#         if use_embeddings:
#             self.embeddings = torch.nn.Linear(input_data_size, embeddings_size, device=device, dtype=dtype)
#             input_data_size = embeddings_size
        
#         generator = [torch.nn.Linear(input_data_size, generator_size[0], device=device, dtype=dtype)]
#         generator.append(activation)
#         for i in range(len(generator_size) -1):
#             generator.append(torch.nn.Linear(generator_size[i], generator_size[i+1], device=device, dtype=dtype))
#             generator.append(activation)
#         self._generator = torch.nn.Sequential(*generator)
        
#         self._weight_head = MemoryShaper(generator_size[-1], target_network_size, device=device, dtype=dtype)

#         self._generate_bias = generate_bias
#         if generate_bias:
#             bias_size = (target_network_size[-1], 1)
#             self._bias_head = MemoryShaper(generator_size[-1], bias_size, device=device, dtype=dtype)
#         else:
#             self._default_bias = torch.zeros((1, generator_size[-1], 1), device=device, dtype=dtype)

#     def forward(self, input_data):
#         if self._use_embeddings:
#             input_data = self.embeddings(input_data)

#         generator_output = self._generator(input_data)

#         # Generate weights
#         weight = self._weight_head(generator_output)
#         # Generate bias if requested else use default
#         if self._generate_bias:
#             bias = self._bias_head(generator_output).squeeze(1)
#         else:
#             bias = self._default_bias.repeat(input_data.shape[0],1)
#         return weight, bias

# class HybridLinearModule(torch.nn.Linear):
#     def __init__(self, in_features: int,
#                  out_features: int,
#                  extra_features: int,
#                  use_embeddings: bool = True,
#                  embeddings_size: int = 32,
#                  generator_size: tuple = (256, 256),
#                  bias: bool = True,
#                  activation: str = "elu",
#                  device=None,
#                  dtype=None,
#         ) -> None:
#         super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

#         # Apply spectral normalization to the weight parameter
#         spectral_norm(self)

#         self.HMRH = HybridMemoryRetrieverHead(
#             input_data_size=extra_features,
#             use_embeddings=use_embeddings,
#             embeddings_size=embeddings_size,
#             generator_size=generator_size,
#             target_network_size=(in_features, out_features),
#             generate_bias=bias,
#             activation=activation,
#             device=device,
#             dtype=dtype,
#         )
            
#     def forward(self, in_features, extra_features):
#         # Explode the tuple of weights
#         weight_scaling, weight_offset, bias_scaling, bias_offset = self.HMRH(extra_features)
#         # Compute the affine transformation of the weights
#         weight = self.weight * weight_scaling + weight_offset
#         bias = self.bias * bias_scaling + bias_offset
#         # Compute the Linear transformation
#         return torch.bmm(weight, in_features.unsqueeze(-1)).squeeze() + bias

# class PureLinearModule(torch.nn.Module):
#     def __init__(self, in_features: int,
#                  out_features: int,
#                  extra_features: int,
#                  use_embeddings: bool = True,
#                  embeddings_size: int = 32,
#                  generator_size: tuple = (256, 256),
#                  bias: bool = True,
#                  activation: str = "elu",
#                  device=None,
#                  dtype=None,
#         ) -> None:
#         super().__init__()

#         self.HMRH = PureMemoryRetrieverHead(
#             input_data_size=extra_features,
#             use_embeddings=use_embeddings,
#             embeddings_size=embeddings_size,
#             generator_size=generator_size,
#             target_network_size=(in_features, out_features),
#             generate_bias=bias,
#             activation=activation,
#             device=device,
#             dtype=dtype,
#         )
            
#     def forward(self, in_features, extra_features):
#         # Explode the tuple of weights
#         weight, bias = self.HMRH(extra_features)
#         # Compute the Linear transformation
#         return torch.bmm(weight, in_features.unsqueeze(-1)).squeeze() + bias

# class HybridMemoryActorNetwork(torch.nn.Module):
#     def __init__(
#             self,
#             num_actor_obs: int = 16,
#             num_memory_obs: int = 12,
#             num_actions: int = 4,
#             actor_hidden_dims = [32, 32],
#             use_embeddings: bool = True,
#             embeddings_size: int = 32,
#             generator_size: tuple = (256, 256),
#             activation = "elu",
#             device=None,
#             dtype=None,
#             **kwargs,
#     ):
#         super().__init__()
#         # Check if activation is a string or a callable
#         if isinstance(activation, str):
#             activation = resolve_nn_activation(activation)

#         self.actor = torch.nn.ModuleList()
        
#         self.actor.append(HybridLinearModule(num_actor_obs,
#                                         actor_hidden_dims[0],
#                                         num_memory_obs,
#                                         use_embeddings=use_embeddings,
#                                         embeddings_size=embeddings_size,
#                                         generator_size=generator_size,
#                                         bias=True,
#                                         activation=activation,
#                                         device=device,
#                                         dtype=dtype))
#         self.actor.append(activation)
#         for i in range(len(actor_hidden_dims)):
#             if i == len(actor_hidden_dims) - 1:
#                 self.actor.append(HybridLinearModule(actor_hidden_dims[i],
#                                                 num_actions,
#                                                 num_memory_obs,
#                                                 use_embeddings=use_embeddings,
#                                                 embeddings_size=embeddings_size,
#                                                 generator_size=generator_size,
#                                                 bias=True,
#                                                 activation=activation,
#                                                 device=device,
#                                                 dtype=dtype))
#             else:
#                 self.actor.append(HybridLinearModule(actor_hidden_dims[i],
#                                                 actor_hidden_dims[i + 1],
#                                                 num_memory_obs,
#                                                 use_embeddings=use_embeddings,
#                                                 embeddings_size=embeddings_size,
#                                                 generator_size=generator_size,
#                                                 bias=True,
#                                                 activation=activation,
#                                                 device=device,
#                                                 dtype=dtype))
#                 self.actor.append(activation)

#     def forward(self, x, memory_data):
#         for act in self.actor:
#             if isinstance(act, HybridLinearModule):
#                 x = act(x, memory_data)
#             else:
#                 x = act(x)
#         return x
    
# class PureMemoryActorNetwork(torch.nn.Module):
#     def __init__(
#             self,
#             num_actor_obs: int = 16,
#             num_memory_obs: int = 12,
#             num_actions: int = 4,
#             actor_hidden_dims = [32, 32],
#             use_embeddings: bool = True,
#             embeddings_size: int = 32,
#             generator_size: tuple = (256, 256),
#             activation = "elu",
#             device=None,
#             dtype=None,
#             **kwargs,
#     ):
#         super().__init__()
#         # Check if activation is a string or a callable
#         if isinstance(activation, str):
#             activation = resolve_nn_activation(activation)

#         self.actor = torch.nn.ModuleList()
        
#         self.actor.append(PureLinearModule(num_actor_obs,
#                                         actor_hidden_dims[0],
#                                         num_memory_obs,
#                                         use_embeddings=use_embeddings,
#                                         embeddings_size=embeddings_size,
#                                         generator_size=generator_size,
#                                         bias=True,
#                                         activation=activation,
#                                         device=device,
#                                         dtype=dtype))
#         self.actor.append(activation)
#         for i in range(len(actor_hidden_dims)):
#             if i == len(actor_hidden_dims) - 1:
#                 self.actor.append(PureLinearModule(actor_hidden_dims[i],
#                                                 num_actions,
#                                                 num_memory_obs,
#                                                 use_embeddings=use_embeddings,
#                                                 embeddings_size=embeddings_size,
#                                                 generator_size=generator_size,
#                                                 bias=True,
#                                                 activation=activation,
#                                                 device=device,
#                                                 dtype=dtype))
#             else:
#                 self.actor.append(PureLinearModule(actor_hidden_dims[i],
#                                                 actor_hidden_dims[i + 1],
#                                                 num_memory_obs,
#                                                 use_embeddings=use_embeddings,
#                                                 embeddings_size=embeddings_size,
#                                                 generator_size=generator_size,
#                                                 bias=True,
#                                                 activation=activation,
#                                                 device=device,
#                                                 dtype=dtype))
#                 self.actor.append(activation)

#     def forward(self, x, memory_data):
#         for act in self.actor:
#             if isinstance(act, PureLinearModule):
#                 x = act(x, memory_data)
#             else:
#                 x = act(x)

#         return x

# if __name__ == "__main__":
#     import torch

#     input_data = torch.randn(1000, 16, device="cuda")
#     memory_data = torch.randn(1000, 12, device="cuda")

#     model = HybridMemoryActorNetwork(
#         num_actor_obs=16,
#         num_memory_obs=12,
#         num_actions=4,
#         actor_hidden_dims=[32, 32],
#         use_embeddings=True,
#         embeddings_size=32,
#         generator_size=(256, 256),
#         activation="elu",
#         device="cuda",
#         dtype=torch.float32,
#     )
#     for i in range(100):
#         output = model(input_data, memory_data)
#     print("actions shape:",output.shape)

#     model = PureMemoryActorNetwork(
#         num_actor_obs=16,
#         num_memory_obs=12,
#         num_actions=4,
#         actor_hidden_dims=[32, 32],
#         use_embeddings=True,
#         embeddings_size=32,
#         generator_size=(256, 256),
#         activation="elu",
#         device="cuda",
#         dtype=torch.float32,
#     )
#     for i in range(100):
#         output = model(input_data, memory_data)
#     print("actions shape:",output.shape)




import torch
import math
from rsl_rl.utils import resolve_nn_activation
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.init as init

class HybridMemoryNetwork:
    def __init__(self):
        pass


class MemoryShaper(torch.nn.Module):
    def __init__(
            self,
            input_size,
            target_size,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__()
        self._K = torch.nn.Linear(input_size, target_size[0], device=device, dtype=dtype)
        self._Q = torch.nn.Linear(input_size, target_size[1], device=device, dtype=dtype)

        # Initialize K and Q with small weights and zero biases
        # This makes their raw outputs small, leading to small 'scores'
        init.normal_(self._K.weight, mean=0.0, std=1e-4) # Small standard deviation
        if self._K.bias is not None:
            init.zeros_(self._K.bias)

        init.normal_(self._Q.weight, mean=0.0, std=1e-4)
        if self._Q.bias is not None:
            init.zeros_(self._Q.bias)

    def forward(self, input_data):
        key = self._K(input_data).unsqueeze(-1)
        query = self._Q(input_data).unsqueeze(-1)
        d_k = query.size(-1)
        scores = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        return scores

class HybridMemoryRetrieverHead(torch.nn.Module):
    def __init__(
            self,
            input_data_size: int = 12,
            use_embeddings: bool = True,
            embeddings_size: int = 32,
            generator_size: tuple = (256,256),
            target_network_size: tuple = (32,32),
            generate_bias: bool = True,
            activation:str = "elu",
            device=None,
            dtype=None,
            **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(activation, str):
            activation = resolve_nn_activation(activation)

        self._use_embeddings = use_embeddings
        if use_embeddings:
            self.embeddings = torch.nn.Linear(input_data_size, embeddings_size, device=device, dtype=dtype)
            # Initialize embeddings layer
            # For ELU, kaiming_uniform_ with a=sqrt(5)
            init.kaiming_uniform_(self.embeddings.weight, a=math.sqrt(5))
            if self.embeddings.bias is not None:
                init.zeros_(self.embeddings.bias)
            input_data_size = embeddings_size
        
        generator = [torch.nn.Linear(input_data_size, generator_size[0], device=device, dtype=dtype)]
        generator.append(activation)
        for i in range(len(generator_size) -1):
            generator.append(torch.nn.Linear(generator_size[i], generator_size[i+1], device=device, dtype=dtype))
            generator.append(activation)
        self._generator = torch.nn.Sequential(*generator)

        # Initialize the generator network
        for module in self._generator.modules():
            if isinstance(module, torch.nn.Linear):
                # Kaiming He initialization for layers followed by ELU
                init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    init.zeros_(module.bias)
        
        self._weight_scaling_head = MemoryShaper(generator_size[-1], target_network_size, device=device, dtype=dtype)
        self._weight_additive_head = MemoryShaper(generator_size[-1], target_network_size, device=device, dtype=dtype)

        self._generate_bias = generate_bias
        if generate_bias:
            bias_size = (target_network_size[-1], 1)
            self._bias_scaling_head = MemoryShaper(generator_size[-1], bias_size, device=device, dtype=dtype)
            self._bias_additive_head = MemoryShaper(generator_size[-1], bias_size, device=device, dtype=dtype)
        else:
            self._default_bias_scaling = torch.ones((1, generator_size[-1]), device=device, dtype=dtype)
            self._default_bias_offset = torch.zeros((1, generator_size[-1]), device=device, dtype=dtype)

    def forward(self, input_data):
        if self._use_embeddings:
            input_data = self.embeddings(input_data)

        generator_output = self._generator(input_data)

        # Generate weights
        weight_scaling = self._weight_scaling_head(generator_output)
        weight_offset = self._weight_additive_head(generator_output)
        # Generate bias if requested else use default
        if self._generate_bias:
            bias_scaling = self._bias_scaling_head(generator_output).squeeze(1)
            bias_offset = self._bias_additive_head(generator_output).squeeze(1)
        else:
            bias_scaling = self._default_bias_scaling.repeat(input_data.shape[0],1)
            bias_offset = self._default_bias_offset.repeat(input_data.shape[0],1)

        # Add 1.0 to the scaling outputs. This ensures they start at 1.0,
        # making the initial modulated weights equal to the base weights.
        # Additive offsets remain as generated (starting near 0)
        weight_scaling = weight_scaling + 1.0
        bias_scaling = bias_scaling + 1.0
        
        return weight_scaling, weight_offset, bias_scaling, bias_offset

class PureMemoryRetrieverHead(torch.nn.Module):
    def __init__(
            self,
            input_data_size: int = 12,
            use_embeddings: bool = True,
            embeddings_size: int = 32,
            generator_size: tuple = (256,256),
            target_network_size: tuple = (32,32),
            generate_bias: bool = True,
            activation:str = "elu",
            device=None,
            dtype=None,
            **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(activation, str):
            activation = resolve_nn_activation(activation)

        self._use_embeddings = use_embeddings
        if use_embeddings:
            self.embeddings = torch.nn.Linear(input_data_size, embeddings_size, device=device, dtype=dtype)
            input_data_size = embeddings_size
        
        generator = [torch.nn.Linear(input_data_size, generator_size[0], device=device, dtype=dtype)]
        generator.append(activation)
        for i in range(len(generator_size) -1):
            generator.append(torch.nn.Linear(generator_size[i], generator_size[i+1], device=device, dtype=dtype))
            generator.append(activation)
        self._generator = torch.nn.Sequential(*generator)
        
        self._weight_head = MemoryShaper(generator_size[-1], target_network_size, device=device, dtype=dtype)

        self._generate_bias = generate_bias
        if generate_bias:
            bias_size = (target_network_size[-1], 1)
            self._bias_head = MemoryShaper(generator_size[-1], bias_size, device=device, dtype=dtype)
        else:
            self._default_bias = torch.zeros((1, generator_size[-1], 1), device=device, dtype=dtype)

    def forward(self, input_data):
        if self._use_embeddings:
            input_data = self.embeddings(input_data)

        generator_output = self._generator(input_data)

        # Generate weights
        weight = self._weight_head(generator_output)
        # Generate bias if requested else use default
        if self._generate_bias:
            bias = self._bias_head(generator_output).squeeze(1)
        else:
            bias = self._default_bias.repeat(input_data.shape[0],1)
        return weight, bias

class HybridLinearModule(torch.nn.Linear):
    def __init__(self, in_features: int,
                 out_features: int,
                 extra_features: int,
                 use_embeddings: bool = True,
                 embeddings_size: int = 32,
                 generator_size: tuple = (256, 256),
                 bias: bool = True,
                 activation: str = "elu",
                 device=None,
                 dtype=None,
        ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

        # Initialize the base weight and bias of the linear layer (IMPORTANT!)
        # These are the parameters that will be modulated.
        # Use Kaiming He for ELU activation.
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # Biases can be initialized to zero or using a small uniform distribution
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound) # or init.zeros_(self.bias)
        

        # Apply spectral normalization to the weight parameter
        # spectral_norm(self)

        self.HMRH = HybridMemoryRetrieverHead(
            input_data_size=extra_features,
            use_embeddings=use_embeddings,
            embeddings_size=embeddings_size,
            generator_size=generator_size,
            target_network_size=(in_features, out_features),
            generate_bias=bias,
            activation=activation,
            device=device,
            dtype=dtype,
        )
            
    def forward(self, in_features, extra_features):
        # Explode the tuple of weights
        weight_scaling, weight_offset, bias_scaling, bias_offset = self.HMRH(extra_features)
        # Compute the affine transformation of the weights
        weight = self.weight * weight_scaling + weight_offset
        bias = self.bias * bias_scaling + bias_offset
        # Compute the Linear transformation
        return torch.bmm(weight, in_features.unsqueeze(-1)).squeeze() + bias

class PureLinearModule(torch.nn.Module):
    def __init__(self, in_features: int,
                 out_features: int,
                 extra_features: int,
                 use_embeddings: bool = True,
                 embeddings_size: int = 32,
                 generator_size: tuple = (256, 256),
                 bias: bool = True,
                 activation: str = "elu",
                 device=None,
                 dtype=None,
        ) -> None:
        super().__init__()

        self.HMRH = PureMemoryRetrieverHead(
            input_data_size=extra_features,
            use_embeddings=use_embeddings,
            embeddings_size=embeddings_size,
            generator_size=generator_size,
            target_network_size=(in_features, out_features),
            generate_bias=bias,
            activation=activation,
            device=device,
            dtype=dtype,
        )
            
    def forward(self, in_features, extra_features):
        # Explode the tuple of weights
        weight, bias = self.HMRH(extra_features)
        # Compute the Linear transformation
        return torch.bmm(weight, in_features.unsqueeze(-1)).squeeze() + bias

class HybridMemoryActorNetwork(torch.nn.Module):
    def __init__(
            self,
            num_actor_obs: int = 16,
            num_memory_obs: int = 12,
            num_actions: int = 4,
            actor_hidden_dims = [32, 32],
            use_embeddings: bool = True,
            embeddings_size: int = 32,
            generator_size: tuple = (256, 256),
            activation = "elu",
            device=None,
            dtype=None,
            **kwargs,
    ):
        super().__init__()
        # Check if activation is a string or a callable
        if isinstance(activation, str):
            activation = resolve_nn_activation(activation)

        self.actor = torch.nn.ModuleList()
        
        self.actor.append(HybridLinearModule(num_actor_obs,
                                        actor_hidden_dims[0],
                                        num_memory_obs,
                                        use_embeddings=use_embeddings,
                                        embeddings_size=embeddings_size,
                                        generator_size=generator_size,
                                        bias=True,
                                        activation=activation,
                                        device=device,
                                        dtype=dtype))
        self.actor.append(activation)
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                self.actor.append(HybridLinearModule(actor_hidden_dims[i],
                                                num_actions,
                                                num_memory_obs,
                                                use_embeddings=use_embeddings,
                                                embeddings_size=embeddings_size,
                                                generator_size=generator_size,
                                                bias=True,
                                                activation=activation,
                                                device=device,
                                                dtype=dtype))
            else:
                self.actor.append(HybridLinearModule(actor_hidden_dims[i],
                                                actor_hidden_dims[i + 1],
                                                num_memory_obs,
                                                use_embeddings=use_embeddings,
                                                embeddings_size=embeddings_size,
                                                generator_size=generator_size,
                                                bias=True,
                                                activation=activation,
                                                device=device,
                                                dtype=dtype))
                self.actor.append(activation)

    def forward(self, x, memory_data):
        for act in self.actor:
            if isinstance(act, HybridLinearModule):
                x = act(x, memory_data)
            else:
                x = act(x)
        return x
    
class PureMemoryActorNetwork(torch.nn.Module):
    def __init__(
            self,
            num_actor_obs: int = 16,
            num_memory_obs: int = 12,
            num_actions: int = 4,
            actor_hidden_dims = [32, 32],
            use_embeddings: bool = True,
            embeddings_size: int = 32,
            generator_size: tuple = (256, 256),
            activation = "elu",
            device=None,
            dtype=None,
            **kwargs,
    ):
        super().__init__()
        # Check if activation is a string or a callable
        if isinstance(activation, str):
            activation = resolve_nn_activation(activation)

        self.actor = torch.nn.ModuleList()
        
        self.actor.append(PureLinearModule(num_actor_obs,
                                        actor_hidden_dims[0],
                                        num_memory_obs,
                                        use_embeddings=use_embeddings,
                                        embeddings_size=embeddings_size,
                                        generator_size=generator_size,
                                        bias=True,
                                        activation=activation,
                                        device=device,
                                        dtype=dtype))
        self.actor.append(activation)
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                self.actor.append(PureLinearModule(actor_hidden_dims[i],
                                                num_actions,
                                                num_memory_obs,
                                                use_embeddings=use_embeddings,
                                                embeddings_size=embeddings_size,
                                                generator_size=generator_size,
                                                bias=True,
                                                activation=activation,
                                                device=device,
                                                dtype=dtype))
            else:
                self.actor.append(PureLinearModule(actor_hidden_dims[i],
                                                actor_hidden_dims[i + 1],
                                                num_memory_obs,
                                                use_embeddings=use_embeddings,
                                                embeddings_size=embeddings_size,
                                                generator_size=generator_size,
                                                bias=True,
                                                activation=activation,
                                                device=device,
                                                dtype=dtype))
                self.actor.append(activation)

    def forward(self, x, memory_data):
        for act in self.actor:
            if isinstance(act, PureLinearModule):
                x = act(x, memory_data)
            else:
                x = act(x)

        return x

if __name__ == "__main__":
    import torch

    input_data = torch.randn(1000, 16, device="cuda")
    memory_data = torch.randn(1000, 12, device="cuda")

    model = HybridMemoryActorNetwork(
        num_actor_obs=16,
        num_memory_obs=12,
        num_actions=4,
        actor_hidden_dims=[32, 32],
        use_embeddings=True,
        embeddings_size=32,
        generator_size=(256, 256),
        activation="elu",
        device="cuda",
        dtype=torch.float32,
    )
    for i in range(100):
        output = model(input_data, memory_data)
    print("actions shape:",output.shape)

    model = PureMemoryActorNetwork(
        num_actor_obs=16,
        num_memory_obs=12,
        num_actions=4,
        actor_hidden_dims=[32, 32],
        use_embeddings=True,
        embeddings_size=32,
        generator_size=(256, 256),
        activation="elu",
        device="cuda",
        dtype=torch.float32,
    )
    for i in range(100):
        output = model(input_data, memory_data)
    print("actions shape:",output.shape)