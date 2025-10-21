import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

def collect_dataset(rollout_episodes: int = 100, seed: int = 42):    
    dataset_path = "cartpole_offline.npz"
    model_path = "ppo_cartpole_expert"

    # Reuse the dataset
    if os.path.exists(dataset_path):
        print(f"Found existing dataset")
        return dataset_path
    
    # If there is no existing dataset, create a new expert model
    env = gym.make("CartPole-v1")   
    env.reset(seed=seed) 
    if os.path.exists(model_path):
        print(f"Found existing model")     
        model = PPO.load(model_path, env=env)   
    else:
        print("Training model")        
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=20000)    
        model.save("ppo_cartpole_expert")

    print("Collecting expert trajectories")

    # Collect expert trajectory of the model
    data = []
    for _ in range(rollout_episodes):
        state, _ = env.reset()
        done, trunc, prev_action = False, False, 0
        traj = []
        while not (done or trunc):
            action, _ = model.predict(state, deterministic=True)
            next_state, reward, done, trunc, _ = env.step(action)
            traj.append((state, prev_action, action, reward))
            state, prev_action = next_state, action
        rtg = 0.0
        processed = []
        # Accumultate rtg backwards
        for state, prev_action, action, reward in reversed(traj):
            rtg += reward
            processed.append((rtg, state, prev_action, action))
        data.extend(reversed(processed))
    env.close()

    # Save the data in .npz file
    np.savez(
        dataset_path,
        rtg=np.array([x[0] for x in data], dtype=np.float32).reshape(-1, 1),
        cartpole_states=np.array([x[1] for x in data], dtype=np.float32),
        prev_actions=np.array([x[2] for x in data], dtype=np.int64),
        actions=np.array([x[3] for x in data], dtype=np.int64),
    )

    print(f"Saved {len(data)} timesteps to {dataset_path}")
    return dataset_path
