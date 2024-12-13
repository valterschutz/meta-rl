class MetaEnv(EnvBase):
    def __init__(self, base_env, base_agent, base_collector_fn, device, seed=None):
        super().__init__(device=device, batch_size=[])
        self.base_env = base_env
        self.base_agent = base_agent

        self.base_collector_fn = base_collector_fn
        # self.base_iter = iter(self.base_collector)

        # Calculate batch size, necessary to know size of observations for meta agent
        i = iter(self.base_collector_fn())
        dummy_td = next(i)
        self.base_batch_size = dummy_td.batch_size.numel()

        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _reset(self, td):
        self.base_agent.reset(mode="train")
        # Reset the base collector
        base_collector = self.base_collector_fn()

        self.base_iter = peekable(base_collector)

        return TensorDict(
            {
                "constant": torch.tensor([42.0], dtype=torch.float32),  # For debugging
                "base_true_mean_reward": torch.tensor([0.0], dtype=torch.float32),
                "base_mean_reward": torch.tensor([0.0], dtype=torch.float32),
                "base_std_reward": torch.tensor([0.0], dtype=torch.float32),
                "last_action": torch.tensor([0], dtype=torch.float32),
                "base": TensorDict(
                    {
                        "losses": TensorDict(
                            {
                                "loss_objective": torch.tensor(
                                    0.0, device=self.device, dtype=torch.float32
                                ),
                                "loss_critic": torch.tensor(
                                    0.0, device=self.device, dtype=torch.float32
                                ),
                                "loss_entropy": torch.tensor(
                                    0.0, device=self.device, dtype=torch.float32
                                ),
                            },
                            batch_size=(),
                        ),
                        "grad_norm": torch.tensor(0.0),
                        "states": F.one_hot(
                            torch.zeros(self.base_batch_size, dtype=torch.long),
                            num_classes=self.base_env.n_states,
                        ).to(torch.float32),
                        "rewards": torch.zeros(
                            (self.base_batch_size, 1), dtype=torch.float32
                        ),
                        "true_rewards": torch.zeros(
                            (self.base_batch_size, 1), dtype=torch.float32
                        ),
                    },
                    batch_size=(),
                ),
                "step": torch.tensor([0.0], dtype=torch.float32),
            },
            batch_size=(),
        )

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _step(self, meta_td, verbose=False):
        """Takes one step in the meta environment, returning the next meta td and also the base td"""

        # Apply meta action, which will affect self.base_iter
        self.base_env.set_constraint_weight(meta_td["action"].item())

        next_meta_td = TensorDict()

        # Get next base batch and update base agent
        base_td = next(self.base_iter)
        next_meta_td["base_true_mean_reward"] = base_td["next", "true_reward"].mean(0)
        next_meta_td["base_mean_reward"] = base_td["next", "reward"].mean(0)
        next_meta_td["base_std_reward"] = base_td["next", "reward"].std(0)
        next_meta_td["last_action"] = meta_td["action"].detach()  # Note detach
        next_meta_td["reward"] = base_td["next", "true_reward"].mean(0)
        base_losses, base_grad_norm = self.base_agent.process_batch(base_td)
        next_meta_td["done"] = not self.is_batches_remaining(self.base_iter)
        next_meta_td["step"] = meta_td["step"] + 1
        next_meta_td["constant"] = meta_td["constant"]

        next_meta_td["base", "states"] = base_td["state"]
        next_meta_td["base", "rewards"] = base_td["next", "reward"]
        next_meta_td["base", "true_rewards"] = base_td["next", "true_reward"]
        next_meta_td["base", "losses"] = base_losses
        next_meta_td["base", "grad_norm"] = base_grad_norm

        return next_meta_td

    @staticmethod
    def is_batches_remaining(peekable_iterator):
        try:
            peekable_iterator.peek()
            return True
        except StopIteration:
            return False

    def _make_spec(self):
        self.observation_spec = Composite(
            constant=Unbounded(shape=(1,), dtype=torch.float32),
            # The state
            base_true_mean_reward=Unbounded(shape=(1,), dtype=torch.float32),
            base_mean_reward=Unbounded(shape=(1,), dtype=torch.float32),
            base_std_reward=Unbounded(shape=(1,), dtype=torch.float32),
            last_action=Bounded(low=0, high=1, shape=(1,), dtype=torch.float32),
            # Base agent that we observe
            base=Composite(
                losses=Composite(
                    loss_objective=Unbounded(shape=(), dtype=torch.float32),
                    loss_critic=Unbounded(shape=(), dtype=torch.float32),
                    loss_entropy=Unbounded(shape=(), dtype=torch.float32),
                    batch_size=(),
                ),
                states=OneHot(
                    self.base_env.n_states,
                    shape=(self.base_batch_size, self.base_env.n_states),
                    dtype=torch.float32,
                ),
                rewards=Unbounded(shape=(self.base_batch_size, 1), dtype=torch.float32),
                true_rewards=Unbounded(
                    shape=(self.base_batch_size, 1), dtype=torch.float32
                ),
                grad_norm=Unbounded(shape=(), dtype=torch.float32),
                batch_size=(),
            ),
            step=Unbounded(shape=(1,), dtype=torch.float32),
            shape=(),
        )
        self.state_spec = Composite(
            constant=Unbounded(shape=(1,), dtype=torch.float32),
            base_true_mean_reward=Unbounded(shape=(1,), dtype=torch.float32),
            base_mean_reward=Unbounded(shape=(1,), dtype=torch.float32),
            base_std_reward=Unbounded(shape=(1,), dtype=torch.float32),
            last_action=Bounded(low=0, high=1, shape=(1,), dtype=torch.float32),
            step=Unbounded(shape=(1,), dtype=torch.float32),
        )
        self.action_spec = Bounded(low=0, high=1, shape=(1,), dtype=torch.float32)
        self.reward_spec = UnboundedContinuous(shape=(1,), dtype=torch.float32)

    # @staticmethod
    # def _meta_state_from_base_td(meta_td, base_td):
    #     # Note the use of .detach() to avoid backpropagating through the base agent
    #     # TODO: detach or not? requires_grad or not?
    #     # TODO: use true_reward or reward?
    #     return TensorDict(
    #         {
    #             "base_mean_reward": base_td["next", "reward"].mean(),
    #             "base_std_reward": base_td["next", "reward"].std(),
    #             "current_weight": meta_td["step"] + 1,
    #         }
    #     )

    # @staticmethod
    # def _meta_reward_from_base_td(meta_td, base_td):
    #     # Note the use of .detach() to avoid backpropagating through the base agent
    #     return base_td["next", "true_reward"].mean()
