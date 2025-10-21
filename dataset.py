import numpy as np
import torch
from torch.utils.data import Dataset


class CartPoleTrajectoryDataset(Dataset):
    """
    Builds (return-to-go, state, prev_action) sequences from the offline CartPole dataset.
    Each sample is a fixed-length chunk with zero padding and an attention mask.
    The `fraction` argument lets you subsample the available expert data.
    """

    def __init__(
        self,
        npz_path: str,
        seq_len: int = 20,
        fraction: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()
        data = np.load(npz_path)

        self.seq_len = seq_len
        self.states = data["cartpole_states"].astype(np.float32)
        self.prev_actions = data["prev_actions"].astype(np.int64)
        self.actions = data["actions"].astype(np.int64)
        self.rtg = data["rtg"].astype(np.float32).reshape(-1)

        # Identify episode boundaries by detecting RTG jumps (RTG resets each episode).
        boundaries = [0]
        for i in range(len(self.rtg) - 1):
            if self.rtg[i + 1] > self.rtg[i]:
                boundaries.append(i + 1)
        boundaries.append(len(self.rtg))

        # Precompute (trajectory_id, start_index) pairs for fast __getitem__ sampling.
        self.trajectories = []
        self.index = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            self.trajectories.append((start, end))
            traj_len = end - start
            for offset in range(traj_len):
                self.index.append((len(self.trajectories) - 1, offset))

        # Optionally subsample the index list to use only a fraction of data.
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in the range (0, 1]")
        if fraction < 1.0:
            rng = np.random.default_rng(seed)
            total = len(self.index)
            keep = max(1, int(total * fraction))
            selected = rng.choice(total, size=keep, replace=False)
            selected.sort()
            self.index = [self.index[i] for i in selected]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        traj_id, offset = self.index[item]
        start, end = self.trajectories[traj_id]
        traj_len = end - start
        horizon = min(self.seq_len, traj_len - offset)

        slc = slice(start + offset, start + offset + horizon)

        rtg = torch.zeros(self.seq_len, 1, dtype=torch.float32)
        states = torch.zeros(self.seq_len, 4, dtype=torch.float32)
        prev_actions = torch.zeros(self.seq_len, dtype=torch.long)
        target_actions = torch.zeros(self.seq_len, dtype=torch.long)
        mask = torch.zeros(self.seq_len, dtype=torch.bool)

        rtg[:horizon, 0] = torch.from_numpy(self.rtg[slc])
        states[:horizon] = torch.from_numpy(self.states[slc])
        prev_actions[:horizon] = torch.from_numpy(self.prev_actions[slc])
        target_actions[:horizon] = torch.from_numpy(self.actions[slc])
        mask[:horizon] = True

        return {
            "rtg": rtg,
            "states": states,
            "prev_actions": prev_actions,
            "actions": target_actions,
            "mask": mask,
        }
