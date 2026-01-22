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

def apply_heatmap(intensity: np.ndarray, colormap_name: str = "viridis") -> np.ndarray:
    """
    Map intensity [0,1] to the requested colormap.
    
    Supported names: viridis, plasma, magma, inferno, jet
    Falls back to viridis if unknown name is given.
    """
    colormaps = {
        "viridis": np.array([
            [0.267, 0.004, 0.329],
            [0.283, 0.130, 0.449],
            [0.263, 0.242, 0.529],
            [0.217, 0.353, 0.554],
            [0.163, 0.456, 0.541],
            [0.126, 0.553, 0.507],
            [0.198, 0.642, 0.447],
            [0.398, 0.722, 0.365],
            [0.669, 0.803, 0.304],
            [0.937, 0.875, 0.274]
        ], dtype=np.float32),
        "plasma": np.array([
            [0.050, 0.012, 0.329],
            [0.282, 0.004, 0.525],
            [0.459, 0.145, 0.604],
            [0.624, 0.267, 0.647],
            [0.780, 0.404, 0.647],
            [0.906, 0.573, 0.604],
            [0.980, 0.757, 0.502],
            [0.992, 0.906, 0.396],
            [0.988, 0.976, 0.271],
            [0.961, 0.992, 0.059]
        ], dtype=np.float32),
        "magma": np.array([
            [0.001, 0.000, 0.014],
            [0.106, 0.016, 0.271],
            [0.255, 0.039, 0.490],
            [0.424, 0.118, 0.553],
            [0.604, 0.235, 0.541],
            [0.780, 0.392, 0.490],
            [0.902, 0.592, 0.396],
            [0.969, 0.765, 0.314],
            [0.988, 0.902, 0.275],
            [0.988, 0.984, 0.396]
        ], dtype=np.float32),
        "inferno": np.array([
            [0.001, 0.000, 0.016],
            [0.110, 0.016, 0.271],
            [0.290, 0.039, 0.490],
            [0.490, 0.118, 0.553],
            [0.690, 0.235, 0.541],
            [0.851, 0.392, 0.490],
            [0.941, 0.592, 0.396],
            [0.984, 0.765, 0.314],
            [0.996, 0.902, 0.275],
            [0.988, 0.984, 0.396]
        ], dtype=np.float32),
        "jet": np.array([
            [0.000, 0.000, 0.500],
            [0.000, 0.000, 1.000],
            [0.000, 1.000, 1.000],
            [0.500, 1.000, 0.500],
            [1.000, 1.000, 0.000],
            [1.000, 0.500, 0.000],
            [1.000, 0.000, 0.000],
            [0.500, 0.000, 0.000],
            [0.250, 0.000, 0.000],
            [0.000, 0.000, 0.000]
        ], dtype=np.float32),
    }
    
    cmap = colormaps.get(colormap_name.lower(), colormaps["viridis"])
        
    # Handle empty or invalid input gracefully
    if intensity.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    idx = np.clip(
        (intensity * (len(cmap) - 1)).astype(int),
        0,
        len(cmap) - 1
    )
    
    return cmap[idx]