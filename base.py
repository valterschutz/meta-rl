"""Helper functions specific to the base environment and base agent"""

from torchrl.collectors import SyncDataCollector
from torchrl.envs.utils import check_env_specs

from agents import BaseAgent
from env import BaseEnv


def get_base_from_config(config):
    """Returns a base environment, base agent, and data collector based on the config"""

    # If batch_size > total_frames, set batch_size to total_frames
    if config.batch_size > config.total_frames:
        batch_size = config.total_frames
    else:
        batch_size = config.batch_size
    # Assuming n_pos is even, calculate x and y
    x, y = BaseEnv.calculate_xy(
        config.n_states,
        config.return_x,
        config.return_y,
        config.big_reward,
        config.gamma,
    )
    # Base env
    env = BaseEnv.get_base_env(
        left_reward=x,
        right_reward=x,
        down_reward=y,
        up_reward=y,
        n_states=config.n_states,
        big_reward=config.big_reward,
        random_start=False,
        punishment=config.punishment,
        seed=None,
        device="cpu",
        constraints_enabled=config.constraints_enabled,
    ).to(config.device)
    check_env_specs(env)

    agent = BaseAgent(
        state_spec=env.state_spec,
        action_spec=env.action_spec,
        num_optim_epochs=config.num_optim_epochs,
        buffer_size=batch_size,
        sub_batch_size=batch_size,
        device=config.device,
        max_grad_norm=config.max_grad_norm,
        lr=config.lr,
        gamma=config.gamma,
        lmbda=config.lmbda,
        clip_epsilon=config.clip_epsilon,
        use_entropy=config.use_entropy,
    )

    collector = SyncDataCollector(
        env,
        agent.policy,
        frames_per_batch=agent.buffer_size,
        total_frames=config.total_frames,
        split_trajs=False,
        device=config.device,
    )

    return env, agent, collector


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
