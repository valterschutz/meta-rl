def get_discrete_sac_loss_module(
    n_states, action_spec, target_entropy, gamma, **agent_kwargs
):
    n_actions = action_spec.n
    # Policy
    actor_net = nn.Sequential(
        nn.Linear(n_states, n_actions),
    )
    policy_module = TensorDictModule(actor_net, in_keys=["state"], out_keys=["logits"])
    policy_module = ProbabilisticActor(
        policy_module,
        spec=action_spec,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        default_interaction_type=InteractionType.RANDOM,
    )

    qvalue_net = nn.Sequential(nn.Linear(n_states, n_actions))
    qvalue_module = ValueOperator(
        qvalue_net,
        in_keys=["state"],
        out_keys=["action_value"],
    )
    use_target_entropy = isinstance(target_entropy, (int, float))
    loss_module = DiscreteSACLoss(
        actor_network=policy_module,
        qvalue_network=qvalue_module,
        action_space=action_spec,
        num_actions=n_actions,
        fixed_alpha=use_target_entropy,
        target_entropy=(target_entropy if use_target_entropy else "auto"),
        loss_function="l2",
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
    return loss_module
