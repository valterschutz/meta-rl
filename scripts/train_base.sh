python train_base.py \
    --env resources/envs#ToyEnv \
    --env-config configs/envs/toy_env.yaml \
    --rl-alg DQN \
    --action-value-network resources/networks#MLP
