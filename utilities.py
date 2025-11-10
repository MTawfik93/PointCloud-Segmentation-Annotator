# utilities.py
import numpy as np

def apply_viridis(intensity: np.ndarray) -> np.ndarray:
    """Map intensity [0,1] to Viridis colormap."""
    VIRIDIS = np.array([
        [0.267,0.004,0.329], [0.283,0.130,0.449], [0.263,0.242,0.529],
        [0.217,0.353,0.554], [0.163,0.456,0.541], [0.126,0.553,0.507],
        [0.198,0.642,0.447], [0.398,0.722,0.365], [0.669,0.803,0.304],
        [0.937,0.875,0.274]
    ], dtype=np.float32)
    idx = np.clip((intensity * (len(VIRIDIS)-1)).astype(int), 0, len(VIRIDIS)-1)
    return VIRIDIS[idx]