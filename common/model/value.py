import torch
import torch.nn as nn
import torch.nn.functional as F

from common.model.activation import FTA
from common.util import check_samples
from common.model.layer import LinearFC, ParallelFC, CompositionalFC, weights_init_

# from activation import FTA
# from layer import LinearFC, ParallelFC, CompositionalFC, weights_init_

class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class ParallelQNetworkBuilder(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dim,
        num_layers,
        num_networks=2,
        num_parallels=1,
        use_layernorm=False,
        use_resnet=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_networks = num_networks
        self.num_total_parallel = num_networks * num_parallels
        self.use_resnet = use_resnet

        in_dim = observation_dim + action_dim + hidden_dim if use_resnet else hidden_dim

        self.l_in = ParallelFC(
            observation_dim + action_dim,
            hidden_dim,
            self.num_total_parallel,
            F.selu,
            use_layernorm,
        )
        self.layers = nn.ModuleList(
            [
                ParallelFC(
                    in_dim, hidden_dim, self.num_total_parallel, F.selu, use_layernorm
                )
                for _ in range(num_layers - 1)
            ]
        )

        self.l_out = ParallelFC(num_parallels * hidden_dim, 1, num_networks)

        self.apply(weights_init_)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)  # [B, SA]
        x = self.l_in(sa)  # [B, n, L] <-- [B, SA]

        if self.use_resnet:
            sa = sa.unsqueeze(1).repeat(1, self.num_total_parallel, 1)  # [B, n, SA]

        for layer in self.layers:
            x = torch.cat([x, sa], dim=2) if self.use_resnet else x
            x = layer(x)

        x = x.reshape(
            x.shape[0], self.num_networks, -1
        )  # [B, Nnet, L*Nparallel] <-- [B, Nnet*Nparallel, L]

        x = self.l_out(x)  # [B, Nnet, 1] <-- [B, Nnet, L*Nparallel]

        return x


class ParallelQNetwork(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dim=64,
        num_layers=2,
        num_networks=2,
        num_parallels=1,
        use_layernorm=False,
        use_resnet=False,
        compile_method="trace",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = ParallelQNetworkBuilder(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_networks=num_networks,
            num_parallels=num_parallels,
            use_resnet=use_resnet,
            use_layernorm=use_layernorm,
        )

        if compile_method == "trace":
            self.model = torch.jit.trace(
                self.model,
                example_inputs=(
                    torch.rand(1, observation_dim),
                    torch.rand(1, action_dim),
                ),
            )
        elif compile_method == "compile":
            self.model = torch.compile(self.model)

    def forward(self, state, action):
        x = self.model(state, action)
        return x

class ParallelMTQNetwork(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        action_dim,
        task_dim,
        hidden_dim=64,
        num_layers=2,
        num_networks=2,
        num_parallels=1,
        use_layernorm=False,
        use_resnet=False,
        compile_method="trace",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = ParallelQNetworkBuilder(
            observation_dim=observation_dim+task_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_networks=num_networks,
            num_parallels=num_parallels,
            use_resnet=use_resnet,
            use_layernorm=use_layernorm,
        )

        if compile_method == "trace":
            self.model = torch.jit.trace(
                self.model,
                example_inputs=(
                    torch.rand(1, observation_dim+task_dim),
                    torch.rand(1, action_dim),
                ),
            )
        elif compile_method == "compile":
            self.model = torch.compile(self.model)

    def forward(self, state, action, task):
        state = torch.cat([state, task], dim=-1)
        x = self.model(state, action)
        return x.squeeze(2)


class ParallelContextualQNetworkBuilder(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        action_dim,
        task_dim,
        hidden_dim,
        context_dim,
        num_layers,
        num_networks=2,
        use_layernorm=False,
        use_resnet=False,
        state_independent_context=False,
    ):
        super().__init__()
        self.num_hidden = hidden_dim
        self.num_layers = num_layers
        self.num_networks = num_networks
        self.context_dim = context_dim
        self.num_total_parallel = num_networks * context_dim
        self.use_resnet = use_resnet
        self.state_independent_context = state_independent_context

        in_dim = observation_dim + action_dim + hidden_dim if use_resnet else hidden_dim

        if self.state_independent_context:
            self.w_in = LinearFC(task_dim, hidden_dim, F.selu, False)
        else:
            self.w_in = LinearFC(observation_dim + task_dim, hidden_dim, F.selu, False)
        self.w_out = LinearFC(hidden_dim, context_dim, F.softmax, False)

        self.l_in = ParallelFC(
            observation_dim + action_dim,
            hidden_dim,
            self.num_total_parallel,
            F.selu,
            use_layernorm,
        )
        self.layers = nn.ModuleList(
            [
                ParallelFC(
                    in_dim, hidden_dim, self.num_total_parallel, F.selu, use_layernorm
                )
                for _ in range(num_layers - 2)
            ]
        )

        self.l_out = ParallelFC(hidden_dim, 1, num_networks)

        self.apply(weights_init_)

    def forward(self, state, action, task):
        if self.state_independent_context:
            w = self.w_in(task) # [B, L] <-- [B, W]
        else:
            w = self.w_in(torch.cat([state, task], dim=1)) # [B, L] <-- [B, S+W]
        w = self.w_out(w) # [B, C] <-- [B, L]

        sa = torch.cat([state, action], 1)  # [B, S+A]
        sa = sa.unsqueeze(1).repeat(1, self.num_total_parallel, 1)  # [B, nNet*C, S+A]
        x = self.l_in(sa)  # [B, nNet*C, L] <-- [B, nNet*C, S+A]

        for layer in self.layers:
            if self.use_resnet:
                x = torch.cat([x, sa], dim=2)  # [B, nNet*C, L+S+A]
            x = layer(x)  # [B, nNet*C, L] <- [B, nNet*C, L+S+A]

        x = x.reshape(x.shape[0], self.num_networks, self.context_dim, self.num_hidden) # [B, nNet, C, L]

        x = torch.einsum('ijkl, ik -> ijl', x, w) # [B, nNet, L] <-- [B, nNet, C, L], [B, C]

        x = self.l_out(x)  # [B, nNet, 1] <-- [B, nNet, L]

        return x.squeeze(2)


class ParallelContextualQNetwork(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        action_dim,
        task_dim,
        hidden_dim,
        context_dim,
        num_layers,
        num_networks=2,
        use_layernorm=False,
        use_resnet=False,
        state_independent_context=False,
        compile_method="trace",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = ParallelContextualQNetworkBuilder(
            observation_dim=observation_dim,
            action_dim=action_dim,
            task_dim=task_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            num_layers=num_layers,
            num_networks=num_networks,
            use_resnet=use_resnet,
            use_layernorm=use_layernorm,
            state_independent_context=state_independent_context,
        )

        if compile_method == "trace":
            self.model = torch.jit.trace(
                self.model,
                example_inputs=(
                    torch.rand(1, observation_dim),
                    torch.rand(1, action_dim),
                    torch.rand(1, task_dim),
                ),
            )
        elif compile_method == "compile":
            self.model = torch.compile(self.model)

    def forward(self, state, action, task):
        x = self.model(state, action, task)
        return x

class CompositionalQNetworkBuilder(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        action_dim,
        task_dim,
        hidden_dim,
        context_dim,
        num_layers,
        num_networks=2,
        use_layernorm=False,
        use_resnet=False,
        use_comp_layer=False,
        state_independent_context=False,
    ):
        super().__init__()
        self.num_networks = num_networks
        self.use_resnet = use_resnet
        self.state_independent_context = state_independent_context

        in_dim = observation_dim + action_dim + hidden_dim if use_resnet else hidden_dim

        if self.state_independent_context:
            self.w_in = LinearFC(task_dim, hidden_dim, F.relu, False)
        else:
            self.w_in = LinearFC(observation_dim + task_dim, hidden_dim, F.relu, False)
        self.w_out = LinearFC(hidden_dim, context_dim, F.softmax, False)

        self.l_in = CompositionalFC(observation_dim + action_dim, hidden_dim, context_dim, F.selu, use_layernorm=use_layernorm, use_comp_layer=use_comp_layer)
        
        self.layers = nn.ModuleList([CompositionalFC(in_dim, hidden_dim, context_dim, F.selu, use_layernorm=use_layernorm, use_comp_layer=use_comp_layer) for _ in range(num_layers-2)])

        self.l_out = ParallelFC(hidden_dim, 1, num_networks)

        self.apply(weights_init_)

    def forward(self, state, action, task):
        if self.state_independent_context:
            w = self.w_in(task) # [B, L] <-- [B, W]
        else:
            w = self.w_in(torch.cat([state, task], dim=1)) # [B, L] <-- [B, S+W]
        w = self.w_out(w) # [B, C] <-- [B, L]

        sa = torch.cat([state, action], 1)  # [B, SA]
        x = self.l_in((sa, w))  # ([B,L], [B,C]) <-- ([B,SA], [B,C])

        for layer in self.layers:
            x = torch.cat([x, sa], dim=1) if self.use_resnet else x # [B, L+S+A] <-- [B,L], [B,S+A]
            x = layer((x, w)) # [B, L], [B,C] <-- ([B,L+S+A], [B,C])

        x = x.unsqueeze(1).repeat(1, self.num_networks, 1)  # [B, nNet, L]
        x = self.l_out(x) # [B, nNet, 1] <-- [B, nNet, L]
        return x.squeeze(2)

class CompositionalQNetwork(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        action_dim,
        task_dim,
        hidden_dim,
        context_dim,
        num_layers,
        num_networks=2,
        use_layernorm=False,
        use_resnet=False,
        use_comp_layer=False,
        state_independent_context=False,
        compile_method="trace",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = CompositionalQNetworkBuilder(
            observation_dim=observation_dim,
            action_dim=action_dim,
            task_dim=task_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            num_layers=num_layers,
            num_networks=num_networks,
            use_resnet=use_resnet,
            use_layernorm=use_layernorm,
            use_comp_layer=use_comp_layer,
            state_independent_context=state_independent_context
        )

        if compile_method == "trace":
            self.model = torch.jit.trace(
                self.model,
                example_inputs=(
                    torch.rand(1, observation_dim),
                    torch.rand(1, action_dim),
                    torch.rand(1, task_dim),
                ),
            )
        elif compile_method == "compile":
            self.model = torch.compile(self.model)

    def forward(self, state, action, task):
        x = self.model(state, action, task)
        return x


class MultiheadSFNetworkBuilder(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        feature_dim,
        action_dim,
        hidden_dim,
        num_layers,
        resnet=False,
        layernorm=False,
        fta=False,
        fta_delta=0.2,
        max_nheads=int(100),
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.max_nheads = max_nheads

        self.feature_dim = feature_dim

        self.resnet = resnet
        self.layernorm = layernorm
        self.fta = fta

        in_dim = observation_dim + action_dim + hidden_dim if resnet else hidden_dim
        out_dim = feature_dim * self.max_nheads

        if self.layernorm:
            self.ln1_1 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_1 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln1_2 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_2 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln1_3 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_3 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln1_4 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_4 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln1_5 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_5 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln1_6 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_6 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln1_7 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_7 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln1_8 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_8 = nn.LayerNorm(in_dim, elementwise_affine=True)

        if fta:
            self.fta_ln = nn.LayerNorm(
                observation_dim + action_dim, elementwise_affine=False
            )
            self.fta = FTA(delta=fta_delta)

            next_dim = (observation_dim + action_dim) * self.fta.nbins
            self.fta_l = nn.Linear(next_dim, observation_dim + action_dim)

        self.l1_1 = nn.Linear(observation_dim + action_dim, hidden_dim)
        self.l2_1 = nn.Linear(observation_dim + action_dim, hidden_dim)

        self.l1_2 = nn.Linear(in_dim, hidden_dim)
        self.l2_2 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 2:
            self.l1_3 = nn.Linear(in_dim, hidden_dim)
            self.l1_4 = nn.Linear(in_dim, hidden_dim)

            self.l2_3 = nn.Linear(in_dim, hidden_dim)
            self.l2_4 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 4:
            self.l1_5 = nn.Linear(in_dim, hidden_dim)
            self.l1_6 = nn.Linear(in_dim, hidden_dim)

            self.l2_5 = nn.Linear(in_dim, hidden_dim)
            self.l2_6 = nn.Linear(in_dim, hidden_dim)

        if num_layers == 8:
            self.l1_7 = nn.Linear(in_dim, hidden_dim)
            self.l1_8 = nn.Linear(in_dim, hidden_dim)

            self.l2_7 = nn.Linear(in_dim, hidden_dim)
            self.l2_8 = nn.Linear(in_dim, hidden_dim)

        self.out1 = nn.Linear(hidden_dim, out_dim)
        self.out2 = nn.Linear(hidden_dim, out_dim)

        self.apply(weights_init_)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        if self.fta:
            sa = self.fta_ln(sa)
            sa = self.fta(sa)
            sa = self.fta_l(sa)

        x1 = F.relu(self.l1_1(sa))
        x2 = F.relu(self.l2_1(sa))

        x1 = torch.cat([x1, sa], dim=1) if self.resnet else x1
        x2 = torch.cat([x2, sa], dim=1) if self.resnet else x2

        x1 = self.ln1_1(x1) if self.layernorm else x1
        x2 = self.ln2_1(x2) if self.layernorm else x2

        x1 = F.relu(self.l1_2(x1))
        x2 = F.relu(self.l2_2(x2))

        if not self.num_layers == 2:
            x1 = torch.cat([x1, sa], dim=1) if self.resnet else x1
            x2 = torch.cat([x2, sa], dim=1) if self.resnet else x2

            x1 = self.ln1_2(x1) if self.layernorm else x1
            x2 = self.ln2_2(x2) if self.layernorm else x2

        if self.num_layers > 2:
            x1 = F.relu(self.l1_3(x1))
            x2 = F.relu(self.l2_3(x2))

            x1 = torch.cat([x1, sa], dim=1) if self.resnet else x1
            x2 = torch.cat([x2, sa], dim=1) if self.resnet else x2

            x1 = self.ln1_3(x1) if self.layernorm else x1
            x2 = self.ln2_3(x2) if self.layernorm else x2

            x1 = F.relu(self.l1_4(x1))
            x2 = F.relu(self.l2_4(x2))
            if not self.num_layers == 4:
                x1 = torch.cat([x1, sa], dim=1) if self.resnet else x1
                x2 = torch.cat([x2, sa], dim=1) if self.resnet else x2

                x1 = self.ln1_4(x1) if self.layernorm else x1
                x2 = self.ln2_4(x2) if self.layernorm else x2

        if self.num_layers > 4:
            x1 = F.relu(self.l1_5(x1))
            x2 = F.relu(self.l2_5(x2))

            x1 = torch.cat([x1, sa], dim=1) if self.resnet else x1
            x2 = torch.cat([x2, sa], dim=1) if self.resnet else x2

            x1 = self.ln1_5(x1) if self.layernorm else x1
            x2 = self.ln2_5(x2) if self.layernorm else x2

            x1 = F.relu(self.l1_6(x1))
            x2 = F.relu(self.l2_6(x2))

            if not self.num_layers == 6:
                x1 = torch.cat([x1, sa], dim=1) if self.resnet else x1
                x2 = torch.cat([x2, sa], dim=1) if self.resnet else x2

                x1 = self.ln1_6(x1) if self.layernorm else x1
                x2 = self.ln2_6(x2) if self.layernorm else x2

        if self.num_layers == 8:
            x1 = F.relu(self.l1_7(x1))
            x2 = F.relu(self.l2_7(x2))

            x1 = torch.cat([x1, sa], dim=1) if self.resnet else x1
            x2 = torch.cat([x2, sa], dim=1) if self.resnet else x2

            x1 = self.ln1_7(x1) if self.layernorm else x1
            x2 = self.ln2_7(x2) if self.layernorm else x2

            x1 = F.relu(self.l1_8(x1))
            x2 = F.relu(self.l2_8(x2))

        x1 = self.out1(x1).view(-1, self.max_nheads, self.feature_dim)
        x2 = self.out2(x2).view(-1, self.max_nheads, self.feature_dim)

        return x1, x2


class MultiheadSFNetwork(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        feature_dim,
        action_dim,
        n_heads,
        hidden_dim=64,
        num_layers=4,
        resnet=False,
        layernorm=False,
        fta=False,
        fta_delta=0.25,
        max_nheads=int(100),
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.max_nheads = max_nheads

        self.model = torch.jit.trace(
            MultiheadSFNetworkBuilder(
                observation_dim=observation_dim,
                feature_dim=feature_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                resnet=resnet,
                layernorm=layernorm,
                fta=fta,
                fta_delta=fta_delta,
                max_nheads=max_nheads,
            ),
            example_inputs=(
                torch.rand(1, observation_dim),
                torch.rand(1, action_dim),
            ),
        )

    def forward(self, state, action):
        n_state = check_samples(state)
        n_action = check_samples(action)
        state = state.view(n_state, -1)
        action = action.view(n_action, -1)

        x1, x2 = self.model(state, action)
        x1 = x1[:, : self.n_heads, :]
        x2 = x2[:, : self.n_heads, :]
        return x1, x2

    def add_head(self, n_heads: int = 1):
        self.n_heads += n_heads
        assert (
            self.n_heads <= self.max_nheads
        ), f"exceed max num heads {self.max_nheads}"


if __name__ == "__main__":
    from torch.profiler import ProfilerActivity, profile, record_function

    obs_dim = 10
    task_dim = 7
    act_dim = 4

    device = "cuda"

    times = 1000
    obs = torch.rand(100, obs_dim).to(device)
    act = torch.rand(100, act_dim).to(device)
    task = torch.rand(100, task_dim).to(device)

    model = CompositionalQNetwork(
        observation_dim=obs_dim,
        action_dim=act_dim,
        task_dim=task_dim,
        hidden_dim=128,
        context_dim=3,
        num_layers=8,
        num_networks=2,
        use_resnet=True,
        use_layernorm=True,
        use_comp_layer=True,
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
                model(obs, act, task)

    print("Compositional")
    print("num parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(prof1.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # model = ParallelContextualQNetwork(
    #     observation_dim=obs_dim,
    #     action_dim=act_dim,
    #     task_dim=task_dim,
    #     hidden_dim=128,
    #     context_dim=3,
    #     num_layers=8,
    #     num_networks=2,
    #     use_resnet=True,
    #     use_layernorm=True,
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
    #             model(obs, act, task)

    # print("Parallel")
    # print("num parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print(prof2.key_averages().table(sort_by="cuda_time_total", row_limit=10))

