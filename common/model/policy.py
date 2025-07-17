import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy 

from common.model.layer import LinearFC, ParallelFC, CompositionalFC, GCNLayer, weights_init_
# from layer import LinearFC, ParallelFC, CompositionalFC, GCNLayer, weights_init_

def repair_checkpoint(path):
    """
    Load a checkpoint and rename keys that start with 'model._orig_mod.' 
    to 'model.' so that they match the expected module names.
    """
    in_state_dict = torch.load(path)
    prefix = "model._orig_mod."
    out_state_dict = {}
    
    for src_key, val in in_state_dict.items():
        if src_key.startswith(prefix):
            # Example: 'model._orig_mod.Theta' -> 'model.Theta'
            dest_key = "model." + src_key[len(prefix):]
        else:
            dest_key = src_key

        out_state_dict[dest_key] = val

    return out_state_dict

class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        try:
            self.load_state_dict(torch.load(path))
        except:
        # Always attempt to 'repair' the checkpoint by removing the '_orig_mod.' prefix
            fixed_state_dict = repair_checkpoint(path)
            self.load_state_dict(fixed_state_dict)


class BaseGaussianPolicyNetwork(BaseNetwork):
    def sample(self, state):
        means, log_stds = self.forward(state)
        normals, xs, actions = self._get_distribution(means, log_stds)
        entropy = self._calc_entropy(normals, xs, actions)
        return actions, entropy, means

    def forward(self, state):
        mean, log_std = self.model(state)
        return mean, log_std

    def _get_distribution(self, means, log_stds):
        stds = log_stds.exp()
        normals = Normal(means, stds)
        xs = normals.rsample()
        actions = torch.tanh(xs)
        return normals, xs, actions

    def _calc_entropy(self, normals, xs, actions, dim=1):
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + self.eps)
        entropy = -log_probs.sum(dim=dim, keepdim=True)
        return entropy


class BaseMultitaskGaussianPolicyNetwork(BaseGaussianPolicyNetwork):
    def sample(self, state, task):
        means, log_stds = self.forward(state, task)
        normals, xs, actions = self._get_distribution(means, log_stds)
        entropy = self._calc_entropy(normals, xs, actions)
        return actions, entropy, means

    def forward(self, state, task):
        mean, log_std = self.model(state, task)
        return mean, log_std



class ParallelGaussianPolicyNetworkBuilder(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dim=32,
        num_layers=2,
        num_parallels=3,
        use_resnet=False,
        use_layernorm=False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.use_resnet = use_resnet
        self.num_parallels = num_parallels

        in_dim = hidden_dim + observation_dim if use_resnet else hidden_dim

        self.l_in = ParallelFC(
            observation_dim, hidden_dim, self.num_parallels, F.relu, use_layernorm
        )

        self.layers = nn.ModuleList(
            [
                ParallelFC(
                    in_dim, hidden_dim, self.num_parallels, F.relu, use_layernorm
                )
                for _ in range(num_layers - 1)
            ]
        )

        self.mean_linear = ParallelFC(num_parallels * hidden_dim, action_dim, 1)
        self.log_std_linear = ParallelFC(num_parallels * hidden_dim, action_dim, 1)

        self.apply(weights_init_)
        nn.init.xavier_uniform_(self.mean_linear.weight, 1e-3)

    def forward(self, state):
        x = self.l_in(state)  # [B, Nparallel, Nhidden] <-- [B, S]

        if self.use_resnet:
            state = state.unsqueeze(1).repeat(1, self.num_parallels, 1)  # [B, Nparallel, S]

        for layer in self.layers:
            x = torch.cat([x, state], dim=2) if self.use_resnet else x
            x = layer(x)

        x = x.reshape(x.shape[0], -1)  # [B, k*Nparallel] <-- [B, Nparallel, k]

        mean = self.mean_linear(x).squeeze(1)  # [B, A]
        log_std = self.log_std_linear(x).squeeze(1)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        return mean, log_std


class ParallelGaussianPolicyNetwork(BaseGaussianPolicyNetwork):
    eps = 1e-6

    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dim=32,
        num_layers=2,
        num_parallels=1,
        use_layernorm=False,
        use_resnet=False,
        compile_method="trace",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = ParallelGaussianPolicyNetworkBuilder(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_parallels=num_parallels,
            use_resnet=use_resnet,
            use_layernorm=use_layernorm,
        )

        if compile_method == "trace":
            self.model = torch.jit.trace(
                self.model,
                example_inputs=(torch.rand(1, observation_dim),),
            )
        elif compile_method == "compile":
            self.model = torch.compile(self.model)


class ParallelMTGaussianPolicyNetwork(BaseGaussianPolicyNetwork):
    eps = 1e-6

    def __init__(
        self,
        observation_dim,
        action_dim,
        task_dim,
        hidden_dim=32,
        num_layers=2,
        num_parallels=1,
        use_layernorm=False,
        use_resnet=False,
        compile_method="trace",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = ParallelGaussianPolicyNetworkBuilder(
            observation_dim=observation_dim+task_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_parallels=num_parallels,
            use_resnet=use_resnet,
            use_layernorm=use_layernorm,
        )

        if compile_method == "trace":
            self.model = torch.jit.trace(
                self.model,
                example_inputs=(torch.rand(1, observation_dim+task_dim),),
            )
        elif compile_method == "compile":
            self.model = torch.compile(self.model)

    def sample(self, state, task):
        means, log_stds = self.forward(state, task)
        normals, xs, actions = self._get_distribution(means, log_stds)
        entropy = self._calc_entropy(normals, xs, actions)
        return actions, entropy, means

    def forward(self, state, task):
        x = torch.cat([state, task], dim=-1)
        mean, log_std = self.model(x)
        return mean, log_std


class ParallelContextualGaussianPolicyNetworkBuilder(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(
        self,
        observation_dim,
        action_dim,
        task_dim,
        context_dim=2,
        hidden_dim=32,
        num_layers=2,
        use_resnet=False,
        use_layernorm=False,
        state_independent_context=False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.use_resnet = use_resnet
        self.state_independent_context = state_independent_context

        in_dim = context_dim + hidden_dim if use_resnet else hidden_dim

        if self.state_independent_context:
            self.w_in = LinearFC(task_dim, hidden_dim, F.relu, False)
        else:
            self.w_in = LinearFC(observation_dim + task_dim, hidden_dim, F.relu, False)
        self.w_out = LinearFC(hidden_dim, context_dim, F.softmax, False)

        self.l_in = ParallelFC(
            observation_dim,
            hidden_dim,
            context_dim,
            F.relu,
            use_layernorm,
        ) 
        self.layers = nn.ModuleList(
            [
                ParallelFC(
                    in_dim, hidden_dim, self.context_dim, F.relu, use_layernorm
                )
                for _ in range(num_layers - 2)
            ]
        )

        self.l_out = LinearFC(hidden_dim, hidden_dim, F.relu, use_layernorm)

        self.mean_linear = LinearFC(hidden_dim, action_dim)
        self.log_std_linear = LinearFC(hidden_dim, action_dim)

        self.apply(weights_init_)
        nn.init.xavier_uniform_(self.mean_linear.weight, 1e-3)

    def forward(self, state, task):
        if self.state_independent_context:
            w = self.w_in(task) # [B, L] <-- [B, W]
        else:
            w = self.w_in(torch.cat([state, task], dim=1)) # [B, L] <-- [B, S+W]
        w = self.w_out(w) # [B, C] <-- [B, L]

        s = state.unsqueeze(1).repeat(1, self.context_dim, 1)  # [B, C, S] <-- [B,S]
        x = self.l_in(s)  # [B, C, L] <- [B, C, S]

        for layer in self.layers:
            if self.use_resnet:
                x = torch.cat([x, s], dim=2)  # [B, C, L+S]
            x = layer(x) # [B, C, L] <-- [B, C, L+S]

        x = torch.einsum("ijk, ij -> ik", x, w)  # [B, L] <- [B, C, L], [B, C]

        x = self.l_out(x)  # [B, L] <-- [B, L]

        mean = self.mean_linear(x)  # [B, A] <-- [B, L]
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        return mean, log_std


class ParallelContextualGaussianPolicyNetwork(BaseMultitaskGaussianPolicyNetwork):
    eps = 1e-6

    def __init__(
        self,
        observation_dim,
        action_dim,
        task_dim,
        context_dim=2,
        hidden_dim=32,
        num_layers=2,
        use_resnet=False,
        use_layernorm=False,
        state_independent_context=False,
        compile_method="trace",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = ParallelContextualGaussianPolicyNetworkBuilder(
            observation_dim=observation_dim,
            action_dim=action_dim,
            task_dim=task_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_resnet=use_resnet,
            use_layernorm=use_layernorm,
            state_independent_context=state_independent_context,
        )

        if compile_method == "trace":
            self.model = torch.jit.trace(
                self.model,
                example_inputs=(
                    torch.rand(1, observation_dim),
                    torch.rand(1, task_dim),
                ),
            )
        elif compile_method == "compile":
            self.model = torch.compile(self.model)
  

class CompositionalGaussianPolicyNetworkBuilder(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(
        self,
        observation_dim,
        action_dim,
        task_dim,
        hidden_dim=32,
        context_dim=16,
        num_layers=2,
        use_resnet=False,
        use_layernorm=False,
        use_comp_layer=False,
        state_independent_context=False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.use_resnet = use_resnet
        self.task_dim = task_dim
        self.state_independent_context = state_independent_context

        in_dim = hidden_dim + observation_dim if use_resnet else hidden_dim

        if self.state_independent_context:
            self.w_in = LinearFC(task_dim, hidden_dim, F.relu, False)
        else:
            self.w_in = LinearFC(observation_dim + task_dim, hidden_dim, F.relu, False)
        self.w_out = LinearFC(hidden_dim, context_dim, F.softmax, False)

        self.l_in = CompositionalFC(
            observation_dim, hidden_dim, context_dim, F.relu, use_layernorm=use_layernorm, use_comp_layer=use_comp_layer
        )

        self.layers = nn.ModuleList(
            [
                CompositionalFC(in_dim, hidden_dim, context_dim, F.relu, use_layernorm=use_layernorm, use_comp_layer=use_comp_layer)
                for _ in range(num_layers - 2)
            ]
        )

        self.mean_linear = CompositionalFC(
            hidden_dim, action_dim, context_dim, use_bias=False, use_comp_layer=False
        )
        self.log_std_linear = CompositionalFC(
            hidden_dim, action_dim, context_dim, use_bias=False, use_comp_layer=False
        )

        self.apply(weights_init_)
        nn.init.xavier_uniform_(self.mean_linear.weight, 1e-3)

    def forward(self, state, task):
        if self.state_independent_context:
            w = self.w_in(task) # [B, L] <-- [B, W]
        else:
            w = self.w_in(torch.cat([state, task], dim=1)) # [B, L] <-- [B, S+W]
        w = self.w_out(w) # [B, C] <-- [B, L]

        x = self.l_in((state, w))  # ([B,L], [B,C]) <-- ([B,S], [B,C])

        for layer in self.layers:
            x = torch.cat([x, state], dim=1) if self.use_resnet else x
            x = layer((x, w))

        mean = self.mean_linear((x, w))  # [B, A] <-- ([B,L], [B,C])
        log_std = self.log_std_linear((x, w))
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        return mean, log_std


class CompositionalGaussianPolicyNetwork(BaseMultitaskGaussianPolicyNetwork):
    eps = 1e-6

    def __init__(
        self,
        observation_dim,
        action_dim,
        task_dim,
        hidden_dim=32,
        context_dim=2,
        num_layers=2,
        use_layernorm=False,
        use_resnet=False,
        use_comp_layer=False,
        state_independent_context=False,
        compile_method="trace",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = CompositionalGaussianPolicyNetworkBuilder(
            observation_dim=observation_dim,
            action_dim=action_dim,
            task_dim=task_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            num_layers=num_layers,
            use_resnet=use_resnet,
            use_layernorm=use_layernorm,
            use_comp_layer=use_comp_layer,
            state_independent_context=state_independent_context,
        )

        if compile_method == "trace":
            self.model = torch.jit.trace(
                self.model,
                example_inputs=(
                    torch.rand(1, observation_dim),
                    torch.rand(1, task_dim),
                ),
            )
        elif compile_method == "compile":
            self.model = torch.compile(self.model)


class GraphGaussianPolicyNetworkBuilder(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(
        self,
        action_dim,
        obs_modal_slice, 
        state_action_adjacency_matrix,
        hidden_dim=16,
        embedding_dim=16,
        num_layers=1,
        num_gcn_layers=1,
        use_layernorm=False,
        use_bias=False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_gcn_layers = num_gcn_layers
        self.hidden_dim = hidden_dim
        self.use_layernorm = use_layernorm

        self.A = nn.Parameter(state_action_adjacency_matrix)

        self.num_action_nodes = action_dim # A

        # modalities
        self.obs_modal_slices = obs_modal_slice 
        self.num_modalities = 0
        for _, modal_type in self.obs_modal_slices:
            self.num_modalities = max(self.num_modalities, modal_type)
        self.num_modalities += 1 # 1's index

        # Embedding layers for nodes
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList([nn.Linear(1, embedding_dim) for _ in range(self.num_modalities)])

        self.gcn_layers = nn.ModuleList(
            [GCNLayer(embedding_dim, hidden_dim, use_layernorm=use_layernorm)]+
            [GCNLayer(hidden_dim, hidden_dim, use_layernorm=use_layernorm) for _ in range(self.num_gcn_layers-1)]
        )

        # Output layers for actions
        self.parallel_layers = nn.ModuleList([
            ParallelFC(hidden_dim, hidden_dim, action_dim, F.relu) for _ in range(num_layers)
        ])

        self.mean_linear = ParallelFC(hidden_dim, 1, action_dim, use_bias=use_bias)
        self.log_std_linear = ParallelFC(hidden_dim, 1, action_dim, use_bias=use_bias)

        self.apply(weights_init_)
        nn.init.xavier_uniform_(self.mean_linear.weight, 1e-3)

    def embed_scalars(self, x, embed_layer):
        # x: [B, X]
        B, X = x.shape
        x_reshaped = x.unsqueeze(-1) # [B,X,1]
        out = embed_layer(x_reshaped.reshape(B*X, 1)) # [B*X, hdim]
        return out.reshape(B, X, self.embedding_dim) # [B,X,hdim]
    
    def forward(self, observation):
        Hs = []
        for modal_slice, modal_type in self.obs_modal_slices:
            modal = observation[:, modal_slice]
            Hs.append(self.embed_scalars(modal, self.embeddings[modal_type]))

        # Concatenate all nodes
        H = torch.cat(Hs, dim=1) # [B, S+G+T+A, hdim]

        A = torch.clip(self.A,-1,1)

        for layer in self.gcn_layers:
            H = layer(H, A)
        
        H_actions = H[:, :self.num_action_nodes, :] # [B,A,hdim]

        x = H_actions
        for layer in self.parallel_layers:
            x = layer(x) # [B, A, hdim] -> [B, A, hdim]

        mean = self.mean_linear(x).squeeze(2)  # [B, A, hdim] -> [B, A, 1] -> [B,A]
        log_std = self.log_std_linear(x).squeeze(2) # [B, A, hdim] -> [B, A, 1] -> [B,A]
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

class GraphGaussianPolicyNetwork(BaseGaussianPolicyNetwork):
    eps = 1e-6

    def __init__(
        self,
        observation_dim,
        action_dim,
        obs_modal_slice,
        state_action_adjacency_matrix,
        hidden_dim=16,
        embedding_dim=16,
        num_layers=1,
        num_gcn_layers=1,
        use_layernorm=False,
        use_bias=False,
        compile_method="trace",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = GraphGaussianPolicyNetworkBuilder(
            action_dim=action_dim,
            obs_modal_slice=obs_modal_slice,
            state_action_adjacency_matrix=state_action_adjacency_matrix,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_gcn_layers=num_gcn_layers,
            use_layernorm=use_layernorm,
            use_bias=use_bias,
        )

        if compile_method == "trace":
            self.model = torch.jit.trace(
                self.model,
                example_inputs=(
                    torch.rand(1, observation_dim),
                ),
            )
        elif compile_method == "compile":
            self.model = torch.compile(self.model)


class MTGraphGaussianPolicyNetworkBuilder(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(
        self,
        action_dim,
        obs_modal_slice, 
        state_action_adjacency_matrix,
        hidden_dim=16,
        embedding_dim=16,
        num_layers=1,
        num_gcn_layers=1,
        use_layernorm=False,
        use_bias=False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_gcn_layers = num_gcn_layers
        self.hidden_dim = hidden_dim
        self.use_layernorm = use_layernorm

        self.A = nn.Parameter(state_action_adjacency_matrix)

        self.num_action_nodes = action_dim # A

        # modalities
        self.obs_modal_slices = obs_modal_slice 
        self.num_modalities = 0
        for _, modal_type in self.obs_modal_slices:
            self.num_modalities = max(self.num_modalities, modal_type)
        self.num_modalities += 2 # 1's index and task is the last modality

        # Embedding layers for nodes
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList([nn.Linear(1, embedding_dim) for _ in range(self.num_modalities)])

        self.gcn_layers = nn.ModuleList(
            [GCNLayer(embedding_dim, hidden_dim, use_layernorm=use_layernorm)]+
            [GCNLayer(hidden_dim, hidden_dim, use_layernorm=use_layernorm) for _ in range(self.num_gcn_layers-1)]
        )

        # Output layers for actions
        self.parallel_layers = nn.ModuleList([
            ParallelFC(hidden_dim, hidden_dim, action_dim, F.relu) for _ in range(num_layers)
        ])

        self.mean_linear = ParallelFC(hidden_dim, 1, action_dim, use_bias=use_bias)
        self.log_std_linear = ParallelFC(hidden_dim, 1, action_dim, use_bias=use_bias)

        self.apply(weights_init_)
        nn.init.xavier_uniform_(self.mean_linear.weight, 1e-3)

    def embed_scalars(self, x, embed_layer):
        # x: [B, X]
        B, X = x.shape
        x_reshaped = x.unsqueeze(-1) # [B,X,1]
        out = embed_layer(x_reshaped.reshape(B*X, 1)) # [B*X, hdim]
        return out.reshape(B, X, self.embedding_dim) # [B,X,hdim]
    
    def forward(self, observation, task):
        Hs = []
        for modal_slice, modal_type in self.obs_modal_slices:
            modal = observation[:, modal_slice]
            Hs.append(self.embed_scalars(modal, self.embeddings[modal_type]))
        Hs.append(self.embed_scalars(task, self.embeddings[-1]))

        # Concatenate all nodes
        H = torch.cat(Hs, dim=1) # [B, S+G+T+A, hdim]

        A = torch.clip(self.A,-1,1)

        for layer in self.gcn_layers:
            H = layer(H, A)
        
        H_actions = H[:, :self.num_action_nodes, :] # [B,A,hdim]

        x = H_actions
        for layer in self.parallel_layers:
            x = layer(x) # [B, A, hdim] -> [B, A, hdim]

        mean = self.mean_linear(x).squeeze(2)  # [B, A, hdim] -> [B, A, 1] -> [B,A]
        log_std = self.log_std_linear(x).squeeze(2) # [B, A, hdim] -> [B, A, 1] -> [B,A]
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

class MTGraphGaussianPolicyNetwork(BaseMultitaskGaussianPolicyNetwork):
    eps = 1e-6

    def __init__(
        self,
        observation_dim,
        task_dim,
        action_dim,
        obs_modal_slice,
        state_action_adjacency_matrix,
        hidden_dim=16,
        embedding_dim=16,
        num_layers=1,
        num_gcn_layers=1,
        use_layernorm=False,
        use_bias=False,
        compile_method="trace",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = MTGraphGaussianPolicyNetworkBuilder(
            action_dim=action_dim,
            obs_modal_slice=obs_modal_slice,
            state_action_adjacency_matrix=state_action_adjacency_matrix,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_gcn_layers=num_gcn_layers,
            use_layernorm=use_layernorm,
            use_bias=use_bias,
        )

        if compile_method == "trace":
            self.model = torch.jit.trace(
                self.model,
                example_inputs=(
                    torch.rand(1, observation_dim),
                    torch.rand(1, task_dim),
                ),
            )
        elif compile_method == "compile":
            self.model = torch.compile(self.model)


if __name__ == "__main__":
    from torch.profiler import ProfilerActivity, profile, record_function

    obs_dim = 13
    act_dim = 3
    task_dim = 3

    device = "cuda"

    times = 100
    obs = torch.rand(100, obs_dim).to(device)
    task = torch.rand(100, task_dim).to(device)

    policy = GraphGaussianPolicyNetwork(
        observation_dim=obs_dim,
        action_dim=act_dim,
        task_dim=task_dim,
        hidden_dim=16,
        num_layers=2,
        use_layernorm=False,
        compile_method="trace",
    ).to(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof1:
        with record_function("model_inference"):
            for _ in range(times):
                policy.sample(obs, task)

    print("Parallel:")
    print("num parameters:", sum(p.numel() for p in policy.parameters() if p.requires_grad))
    print(prof1.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # policy = CompositionalGaussianPolicyNetwork(
    #     observation_dim=obs_dim,
    #     action_dim=act_dim,
    #     task_dim=task_dim,
    #     context_dim=3,
    #     hidden_dim=8,
    #     num_layers=2,
    #     use_resnet=False,
    #     use_layernorm=False,
    #     use_comp_layer=True,
    #     compile_method="trace",
    # ).to(device)

    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    # ) as prof2:
    #     with record_function("model_inference"):
    #         for _ in range(times):
    #             policy.sample(obs, task)

    # print("Compositional:")
    # print("num parameters:", sum(p.numel() for p in policy.parameters() if p.requires_grad))
    # print(prof2.key_averages().table(sort_by="cuda_time_total", row_limit=10))
