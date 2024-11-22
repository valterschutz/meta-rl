class BaseAgent:
    def __init__(
        self,
        state_spec,
        action_spec,
        num_optim_epochs,
        buffer_size,
        sub_batch_size,
        device,
    ):
        # TODO: calculate number of states and actions from specs
        # Base agent networks
        self.hidden_units = 4
        self.actor_net = nn.Sequential(
            OneHotLayer(num_classes=n_states),
            nn.Linear(n_states, self.hidden_units),
            nn.Tanh(),
            nn.Linear(self.hidden_units, n_actions),
        ).to(device)
        self.policy = TensorDictModule(
            self.actor_net, in_keys=["pos"], out_keys=["logits"]
        )
        self.policy = ProbabilisticActor(
            module=self.policy,
            spec=action_spec,
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=True,
        )
        self.value_net = nn.Sequential(
            OneHotLayer(num_classes=n_states),
            nn.Linear(n_states, self.hidden_units),
            nn.Tanh(),
            nn.Linear(self.hidden_units, 1),
        ).to(device)
        self.value_module = ValueOperator(
            self.value_net, in_keys=["pos"], out_keys=["state_value"]
        )
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.value_module,
            clip_epsilon=clip_epsilon,
            entropy_bonus=False,
        )
        self.advantage_module = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=self.value_module,
        )
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=lr)

        self.num_optim_epochs = num_optim_epochs

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size, device=device),
            sampler=SamplerWithoutReplacement(),
        )

        self.sub_batch_size = sub_batch_size

    def process_batch(td):
        # Process a single batch of data
        for _ in range(self.num_optim_epochs):
            self.advantage_module(td)
            self.replay_buffer.extend(td)
            for _ in range(len(td) // self.sub_batch_size):
                sub_td = self.replay_buffer.sample(self.sub_batch_size)
                loss = self.loss_module(sub_td)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()


class MetaAgent:
    def __init__(self, state_spec, action_spec, device):
        self.hidden_units = 16
        self.actor_net = nn.Sequential(
            nn.Linear(2, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1),
            nn.Sigmoid(),  # Interpret action as probability to set constraint
        ).to(device)
        policy = TensorDictModule(actor_net, in_keys=["state"], out_keys=["probs"])
        self.policy = ProbabilisticActor(
            module=policy,
            spec=action_spec,
            in_keys=["probs"],
            distribution_class=Bernoulli,
            return_log_prob=True,
        )
        self.value_net = nn.Sequential(
            nn.Linear(2, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1),
        ).to(device)
        self.value_module = ValueOperator(
            value_net, in_keys=["state"], out_keys=["state_value"]
        )
        self.loss_module = ClipPPOLoss(
            actor_network=policy,
            critic_network=value_module,
            clip_epsilon=clip_epsilon,
            entropy_bonus=False,
        )
        self.advantage_module = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=value_module,
        )
        optim = torch.optim.Adam(self.loss_module.parameters(), lr=lr)

    def process_batch(td):
        pass
