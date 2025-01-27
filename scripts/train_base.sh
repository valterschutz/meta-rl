python train_base.py \
    --env resources/envs#get_toy_env \
    --env-config configs/envs/jan27_env.yaml \
    --rl-alg DQN \
    --action-value-network resources/networks#jan27_qnet \
    --alg-config configs/algs/jan27_dqn.yaml \
    --agent-config configs/agents/jan27_agent.yaml \
    --trainer-config configs/trainers/jan27_trainer.yaml
