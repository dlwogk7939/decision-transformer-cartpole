import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CartPoleTrajectoryDataset
from decision_transformer import DecisionTransformer
from generate_expert_data import generate_dataset

def get_device():      
    if torch.cuda.is_available():
        print("Using NVIDIA GPU")
        return torch.device("cuda")         # NVIDIA GPU
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using Apple Silicon GPU")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")          # Apple Silicon GPU    
    else:
        print("No GPU found, using CPU")
        return torch.device("cpu")
    
# Return hidden_size, n_layer, n_head according to the transformer_size
def get_transformer_size(transformer_size: int = 1):
    if transformer_size == 0:               # small
        return (64, 2, 2)
    elif transformer_size == 1:             # medium
        return (128, 3, 2)
    else:                                   # large
        return (256, 4, 4)

def train_transformer(
    npz_path: str,
    seq_len: int = 20,
    batch_size: int = 64,
    epochs: int = 5,
    lr: float = 1e-4,
    # hidden_size: int = 128,
    # n_layer: int = 3,
    # n_head: int = 2,
    transformer_size: int = 1,
    input_fraction: float = 1.0,
    seed: int = 42,
):
    dataset = CartPoleTrajectoryDataset(
        npz_path=npz_path,
        seq_len=seq_len,
        fraction=input_fraction,
        seed=seed,
    )

    g = torch.Generator()
    g.manual_seed(seed)    

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
    
    device = get_device()
    hidden_size, n_layer, n_head = get_transformer_size(transformer_size)

    model = DecisionTransformer(
        hidden_size=hidden_size,
        n_layer=n_layer,
        n_head=n_head,
        seq_len=seq_len,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("Training transformer")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_tokens = 0
        for batch in loader:
            rtg = batch["rtg"].to(device)
            states = batch["states"].to(device)
            prev_actions = batch["prev_actions"].to(device)
            target_actions = batch["actions"].to(device)
            mask = batch["mask"].to(device)

            logits = model(rtg, states, prev_actions, padding_mask=mask)
            loss = criterion(logits[mask], target_actions[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

        print(f"Epoch {epoch + 1}/{epochs} - avg loss: {total_loss / max(total_tokens, 1):.4f}")

    return model


def evaluate_transformer(
    model: DecisionTransformer,
    env_name: str = "CartPole-v1",
    episodes: int = 100,
    target_return: float = 500.0,
):
    device = next(model.parameters()).device
    seq_len = model.seq_len
    state_dim = model.state_dim

    env = gym.make(env_name)
    model.eval()
    returns = []

    with torch.no_grad():
        for _ in range(episodes):
            state, _ = env.reset()
            state = np.asarray(state, dtype=np.float32)

            state_hist = [state]
            prev_action_hist = [0]
            running_return = target_return
            rtg_hist = [running_return]
            episode_return = 0.0
            done = False

            while not done:
                hist_len = len(state_hist)
                take = min(hist_len, seq_len)
                state_seq = torch.zeros(seq_len, state_dim, device=device)
                prev_action_seq = torch.zeros(seq_len, dtype=torch.long, device=device)
                rtg_seq = torch.zeros(seq_len, 1, device=device)

                state_window = np.asarray(state_hist[-take:], dtype=np.float32)
                prev_action_window = np.asarray(prev_action_hist[-take:], dtype=np.int64)
                rtg_window = np.asarray(rtg_hist[-take:], dtype=np.float32)

                state_seq[-take:] = torch.from_numpy(state_window).to(device)
                prev_action_seq[-take:] = torch.from_numpy(prev_action_window).to(device)
                rtg_seq[-take:, 0] = torch.from_numpy(rtg_window).to(device)

                padding_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
                padding_mask[-take:] = True

                logits = model(
                    rtg_seq.unsqueeze(0),
                    state_seq.unsqueeze(0),
                    prev_action_seq.unsqueeze(0),
                    padding_mask=padding_mask.unsqueeze(0),
                )
                action = torch.argmax(logits[0, -1]).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                running_return = max(running_return - reward, 0.0)
                done = terminated or truncated

                if not done:
                    next_state = np.asarray(next_state, dtype=np.float32)
                    state_hist.append(next_state)
                    prev_action_hist.append(action)
                    rtg_hist.append(running_return)

            returns.append(episode_return)

    env.close()

    avg_return = float(np.mean(returns)) if returns else 0.0
    std_return = float(np.std(returns)) if returns else 0.0
    return avg_return, std_return, returns


def main(    
    rollout_episodes: int = 100,
    input_fraction: float = 1.0,
    epochs: int = 5,
    transformer_size = 1,
    seed: int = 42,
):    
    dataset_path = generate_dataset(rollout_episodes=rollout_episodes, seed=seed)
    model = train_transformer(
        dataset_path,
        transformer_size = transformer_size,
        epochs = epochs,
        input_fraction=input_fraction,
        seed=seed,
    )
    avg_return, std_return, returns = evaluate_transformer(model)
    print(f"Decision Transformer average return: {avg_return:.2f} ± {std_return:.2f} over {len(returns)} episodes")

if __name__ == "__main__":    
    import argparse
    parser = argparse.ArgumentParser(description="CartPole Decision Transformer")
    parser.add_argument("--rollout_episodes", type=int, default=100)
    parser.add_argument("--input_fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--transformer_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        rollout_episodes=args.rollout_episodes,
        input_fraction=args.input_fraction,
        epochs=args.epochs,
        transformer_size=args.transformer_size,
        seed=args.seed,
    )
