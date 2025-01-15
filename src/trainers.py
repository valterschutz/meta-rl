
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict import TensorDict, TensorDictBase
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn.probabilistic import InteractionType
from torch import multiprocessing, nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, BoundedTensorSpec
from torchrl.data.replay_buffers import (ListStorage, PrioritizedSampler,
                                         ReplayBuffer)
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DMControlEnv, DoubleToFloat,
                          ObservationNorm, ParallelEnv, StepCounter,
                          TransformedEnv, set_gym_backend)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import CatTensors, Transform, RenameTransform, DoubleToFloat, DTypeCastTransform
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import (ExplorationType, check_env_specs,
                                set_exploration_type)
from torchrl.modules import (OneHotCategorical, ProbabilisticActor, TanhNormal,
                             ValueOperator)
from torchrl.objectives import DiscreteSACLoss, SoftUpdate, ValueEstimators
from torchrl.objectives.value import GAE
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger
from tqdm import tqdm

import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), "./"))

from envs.toy_env import ToyEnv
from envs.dm_env import ConstraintDMControlEnv
from utils import calc_return
from agents.base_agents import ToySACAgent, PointSACAgent, ReacherSACAgent


# TODO: rewrite to use OffpolicyTrainer
def train_toy_base_agent(device, total_frames, min_buffer_size, n_states, big_reward, gamma, shortcut_steps, return_x, return_y, when_constraints_active, times_to_eval, log, progress_bar, batch_size, sub_batch_size, num_epochs, agent_alg):
    """
    Train a base agent in the toy environment.
    """
    env_max_steps = 5*n_states

    n_batches = total_frames // batch_size
    eval_every_n_batch = n_batches // times_to_eval

    transforms = Compose(
        StepCounter(max_steps=env_max_steps),
        DTypeCastTransform(dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"])
    )
    x, y = ToyEnv.calculate_xy(n_states=n_states, shortcut_steps=shortcut_steps, return_x=return_x, return_y=return_y, big_reward=big_reward, punishment=0.0, gamma=gamma)
    env = ToyEnv(
        left_reward=x,
        right_reward=x,
        down_reward=y,
        up_reward=y,
        n_states=n_states,
        shortcut_steps=shortcut_steps,
        big_reward=big_reward,
        punishment=0.0,
        constraints_active=False,
        random_start=False,
        seed=None,
        device=device
    )
    env = TransformedEnv(
        env,
        transforms
    )

    if agent_alg == "SAC":
        agent = ToySACAgent(
            state_spec=env.state_spec,
            n_states = n_states,
            action_spec=env.action_spec,
            device=device,
            buffer_size = total_frames,
            min_buffer_size = min_buffer_size,
            batch_size = batch_size,
            sub_batch_size = sub_batch_size,
            num_epochs = num_epochs,
            gamma = gamma,
        )
    else:
        raise ValueError(f"Unknown agent algorithm {agent_alg}")

    rand_collector = SyncDataCollector(
        env,
        None,
        frames_per_batch=agent.batch_size,
        total_frames=agent.min_buffer_size,
        split_trajs=False,
        device=device,
    )

    collector = SyncDataCollector(
        env,
        agent.policy_module,
        frames_per_batch=agent.batch_size,
        total_frames=total_frames-agent.min_buffer_size,
        split_trajs=False,
        device=device,
    )

    if progress_bar:
        pbar = tqdm(total=total_frames)
    if isinstance(when_constraints_active, float):
        batch_to_activate_constraints = int(n_batches * when_constraints_active)
    for td in rand_collector:
        agent.process_batch(td, constraints_active=False)
        if progress_bar:
            pbar.update(td.numel())

    eval_returns = []
    try:
        for i, td in enumerate(collector):
            # td["action"] = td["action"].to(torch.float32) # Due to bug in torchrl, need to manually cast to float
            collector.update_policy_weights_() # Check if this is necessary

            # Constraints are either deterministically set at some batch or decided by a callback function
            if isinstance(when_constraints_active, float):
                constraints_active: bool = i >= batch_to_activate_constraints
            elif callable(when_constraints_active):
                constraints_active: bool = when_constraints_active(td)

            loss_dict, info_dict = agent.process_batch(td, constraints_active=constraints_active)

            if log:
                wandb.log({
                    "normal_reward": td["next", "normal_reward"].mean().item(),
                    "constraint_reward": td["next", "constraint_reward"].mean().item(),
                    "reward": (td["next", "normal_reward"] + td["next", "constraint_reward"]).mean().item(),
                    "max step count": td["step_count"].max().item(),
                    **loss_dict,
                    **info_dict,
                    "batch": i,
                    # "next state distribution": wandb.Histogram(td["next","observation"].argmax(dim=-1).cpu()+1),
                    "next state distribution": wandb.Histogram(td["next","observation"].cpu()+1),
                    "action distribution": wandb.Histogram(td["action"].argmax(dim=-1).cpu()+1),
                    "policy 'norm'": sum((p**2).sum() for p in agent.loss_module.actor_network_params.parameters()),
                    "value 'norm'": sum((p**2).sum() for p in agent.loss_module.qvalue_network_params.parameters()),
                    "constraints_active": float(constraints_active)
                })
            if i % eval_every_n_batch == 0:
                with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
                    eval_data = env.rollout(1000, agent.policy_module)
                # Always use constrained return for evaluation
                eval_return = calc_return((eval_data["next", "normal_reward"]+eval_data["next","constraint_reward"]).flatten(), gamma)
                if log:
                    wandb.log({
                        "eval return": eval_return,
                        "batch": i
                    })
                eval_returns.append(eval_return)

            if progress_bar:
                pbar.update(td.numel())
    except KeyboardInterrupt as e:
        print(f"Training interrupted due to an error: {e}")
        if progress_bar:
            pbar.close()
    return eval_returns


class OffpolicyTrainer():
    def __init__(self, env, agent, progress_bar, times_to_eval, collector_device, log, max_eval_steps, collector_args, env_gamma, eval_env=None):
        self.env = env
        self.agent = agent
        self.total_frames = collector_args["total_frames"]
        self.n_batches = self.total_frames // collector_args["batch_size"]
        self.eval_every_n_batch = self.n_batches // times_to_eval
        self.log = log
        self.max_eval_steps = max_eval_steps
        self.progress_bar = progress_bar
        self.env_gamma = env_gamma
        if eval_env is None:
            self.eval_env = env
        else:
            self.eval_env = eval_env

        self.rand_collector = SyncDataCollector(
            env,
            None,
            frames_per_batch=collector_args["batch_size"],
            total_frames=agent.min_buffer_size,
            split_trajs=False,
            device=collector_device,
        )

        self.collector = SyncDataCollector(
            env,
            agent.train_policy,
            frames_per_batch=collector_args["batch_size"],
            total_frames=self.total_frames-agent.min_buffer_size,
            split_trajs=False,
            device=collector_device,
        )

    def train(self, when_constraints_active):
        if self.progress_bar:
            pbar = tqdm(total=self.total_frames)
        if isinstance(when_constraints_active, float):
            batch_to_activate_constraints = int(self.n_batches * when_constraints_active)
        for td in self.rand_collector:
            self.agent.process_batch(td, constraints_active=False)
            if self.progress_bar:
                pbar.update(td.numel())

        eval_true_returns = []
        try:
            for i, td in enumerate(self.collector):
                self.collector.update_policy_weights_()

                # Constraints are either deterministically set at some batch or decided by a callback function
                if isinstance(when_constraints_active, float):
                    constraints_active: bool = i >= batch_to_activate_constraints
                elif callable(when_constraints_active):
                    constraints_active: bool = when_constraints_active(td)

                loss_dict, info_dict = self.agent.process_batch(td, constraints_active=constraints_active)

                if self.log:
                    wandb.log({
                        "normal_reward": td["next", "normal_reward"].mean().item(),
                        "constraint_reward": td["next", "constraint_reward"].mean().item(),
                        "true_reward": (td["next", "normal_reward"] + td["next", "constraint_reward"]).mean().item(),
                        "batch": i,
                        **loss_dict,
                        **info_dict,
                    })
                if i % self.eval_every_n_batch == 0:
                    with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
                        eval_data = self.eval_env.rollout(self.max_eval_steps, self.agent.eval_policy)
                    # Always use constrained return for evaluation
                    eval_normal_return = calc_return((eval_data["next", "normal_reward"]).flatten(), self.env_gamma)
                    eval_true_return = calc_return((eval_data["next", "normal_reward"]+eval_data["next","constraint_reward"]).flatten(), self.env_gamma)
                    if self.log:
                        wandb.log({
                            "eval normal return": eval_normal_return,
                            "eval true return": eval_true_return,
                            "batch": i
                        })
                    eval_true_returns.append(eval_true_return)
                    # If evaluation env is pixelated, record video
                    if "pixels" in eval_data:
                        wandb.log({"video": wandb.Video(eval_data["pixels"].permute(0, 3, 1, 2).cpu(), fps=30)})

                if self.progress_bar:
                    pbar.update(td.numel())
        except KeyboardInterrupt as e:
            print(f"Training interrupted.")
            if self.progress_bar:
                pbar.close()
        return eval_true_returns

def log_video(td, i):
    """
    Logs a video to wandb, assuming that the "pixels" entry is present in `td`.
    """
    # dt = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    # exp_name = f"{env_type}|{agent_type}_{dt}"
    # logger = CSVLogger(exp_name=exp_name, log_dir="logs", video_format="pt")
    # recorder = VideoRecorder(logger, tag="temp")
    # record_env = TransformedEnv(pixel_env, recorder)
    # with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
    #     rollout = record_env.rollout(max_steps=sys.maxsize, policy=trainer.loss_module.actor_network)
    # recorder.dump() # Saves video as a .pt file at `csv`
    # # Now load the file as a tensor
    # video = torch.load(Path(logger.log_dir) / exp_name / "videos" / "temp_0.pt").numpy()
    wandb.log({"video": wandb.Video(video)})

# TODO: not finished
def train_meta_agent(
    device,
    n_base_episodes,
    log,
    progress_bar,
    batch_size,
    sub_batch_size,
    num_epochs
):
    meta_agent = MetaAgent(state_keys, device, buffer_size, min_buffer_size, sub_batch_size, num_epochs, lr, gamma, target_eps, alpha, beta, max_grad_norm)
    # Define the meta callback, a function that takes the base agents tensordict and returns a boolean that indicates if
    # the constraints should be activated
    def meta_callback(base_td):
        return True
    eval_returns = []
    try:
        for i in range(n_base_episodes):
            base_returns = train_base_agent(
                device=torch.device("cpu"),
                total_frames=50_000,
                min_buffer_size=0,
                n_states=n_states,
                shortcut_steps=5,
                return_x=return_x,
                return_y=-100,
                when_constraints_active=meta_callback,
                times_to_eval=20,
                log=True,
                progress_bar=True,
                batch_size = 200,
                sub_batch_size = 20,
                num_epochs = 100
            )

            if log:
                wandb.log({
                    "reward": td["next", "reward"].mean().item(),
                    "max step count": td["step_count"].max().item(),
                    **loss_dict,
                    **info_dict,
                    "batch": i,
                    "state distribution": wandb.Histogram(td["state"].argmax(dim=-1).cpu()),
                    "action distribution": wandb.Histogram(td["action"].argmax(dim=-1).cpu()),
                    "policy 'norm'": sum((p**2).sum() for p in agent.policy_module.parameters()),
                    "when_constraints_active": when_constraints_active if isinstance(when_constraints_active, float) else 0.0,
                })
            if i % eval_every_n_batch == 0:
                with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
                    eval_data = env.rollout(100, agent.policy_module)
                # Always use constrained return for evaluation
                eval_return = calc_return((eval_data["next", "normal_reward"]+eval_data["next","constraint_reward"]).flatten(), gamma)
                if log:
                    wandb.log({
                        "eval return": eval_return,
                        "batch": i
                    })
                eval_returns.append(eval_return)

            if progress_bar:
                pbar.update(td.numel())
    except Exception as e:
        print(f"Training interrupted due to an error: {e}")
        if progress_bar:
            pbar.close()
    return eval_returns
