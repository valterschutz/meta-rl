"""Helper functions specific to the base environment and base agent"""

from torchrl.collectors import SyncDataCollector
from torchrl.envs.utils import check_env_specs

from agents import BaseAgent
from env import BaseEnv


def get_base_from_config(config):
    """Returns a base environment, base agent, and data collector based on the config"""

    agent = PPOBaseAgent(
        state_spec=env.state_spec,
        action_spec=env.action_spec,
        num_optim_epochs=config.num_optim_epochs,
        buffer_size=config.batch_size,  # Same as batch_size due to PPO being an on-policy method
        sub_batch_size=config.sub_batch_size,
        device=config.device,
        max_grad_norm=config.max_grad_norm,
        lr=config.lr,
        gamma=config.gamma,
        lmbda=config.lmbda,
        clip_epsilon=config.clip_epsilon,
        use_entropy=config.use_entropy,
        entropy_coef=config.entropy_coef,
        critic_coef=config.critic_coef,
    )

    collector_fn = lambda: SyncDataCollector(
        env,
        agent.policy,
        frames_per_batch=config.batch_size,
        total_frames=config.total_frames,
        split_trajs=False,
        device=config.device,
    )

    return env, agent, collector_fn


def print_base_rollout(td, gamma):
    """Prints visited states, actions taken, and rewards received in a base rollout"""

    G = 0
    print("<step>: (<state>, <action>, <next_reward>)")
    for i in range(td["step_count"].max().item() + 1):
        G += td["next", "reward"][i].item() * gamma**i
        print(
            f"{i}: ({td['state'][i].item()}, {td['action'][i].item()}, {td['next', 'reward'][i].item()})"
        )
    print(f"Return: {G}")
    print()