import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud(points, colors, intensity, subsample=None):
    """
    Visualize 3D point cloud with grayscale colors from intensity.
    
    Args:
        points: Nx3 array of xyz coordinates
        colors: Nx3 array of RGB colors (0-255 range)
        intensity: N array of intensity values
        subsample: Optional, subsample points for faster visualization (e.g., 10000)
    """
    # Subsample if needed for performance
    if subsample and len(points) > subsample:
        indices = np.random.choice(len(points), subsample, replace=False)
        points_vis = points[indices]
        colors_vis = colors[indices]
        intensity_vis = intensity[indices]
    else:
        points_vis = points
        colors_vis = colors
        intensity_vis = intensity
    
    # Convert colors from 0-255 to 0-1 range for matplotlib
    colors_normalized = colors_vis.astype(np.float32) / 255.0
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(221, projection='3d')
    scatter = ax1.scatter(points_vis[:, 0], 
                         points_vis[:, 1], 
                         points_vis[:, 2],
                         c=colors_normalized,
                         s=1,
                         alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'3D Point Cloud ({len(points_vis):,} points)')
    
    # Set equal aspect ratio
    max_range = np.array([points_vis[:, 0].max() - points_vis[:, 0].min(),
                          points_vis[:, 1].max() - points_vis[:, 1].min(),
                          points_vis[:, 2].max() - points_vis[:, 2].min()]).max() / 2.0
    
    mid_x = (points_vis[:, 0].max() + points_vis[:, 0].min()) * 0.5
    mid_y = (points_vis[:, 1].max() + points_vis[:, 1].min()) * 0.5
    mid_z = (points_vis[:, 2].max() + points_vis[:, 2].min()) * 0.5
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # XY view (top-down)
    ax2 = fig.add_subplot(222)
    ax2.scatter(points_vis[:, 0], points_vis[:, 1], 
                c=colors_normalized, s=0.5, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top View (XY)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # XZ view (front)
    ax3 = fig.add_subplot(223)
    ax3.scatter(points_vis[:, 0], points_vis[:, 2], 
                c=colors_normalized, s=0.5, alpha=0.6)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('Front View (XZ)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # YZ view (side)
    ax4 = fig.add_subplot(224)
    ax4.scatter(points_vis[:, 1], points_vis[:, 2], 
                c=colors_normalized, s=0.5, alpha=0.6)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('Side View (YZ)')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_intensity_histogram(intensity):
    """
    Plot histogram of intensity values.
    
    Args:
        intensity: N array of intensity values
    """
    plt.figure(figsize=(10, 6))
    plt.hist(intensity, bins=100, color='gray', edgecolor='black', alpha=0.7)
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.title('Intensity Distribution')
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_with_open3d(points, colors):
    """
    Visualize using Open3D (if available) for better interactive experience.
    
    Args:
        points: Nx3 array of xyz coordinates
        colors: Nx3 array of RGB colors (0-255 range)
    """
    try:
        import open3d as o3d
        
        # Create point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Convert colors from 0-255 to 0-1 range
        colors_normalized = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
        
        # Visualize
        print("Opening Open3D visualizer...")
        print("Controls:")
        print("  - Mouse left: Rotate")
        print("  - Mouse right: Pan")
        print("  - Mouse wheel: Zoom")
        print("  - Press 'H' for help")
        
        o3d.visualization.draw_geometries([pcd],
                                         window_name='Point Cloud Viewer',
                                         width=1200,
                                         height=800,
                                         left=50,
                                         top=50,
                                         point_show_normal=False)
    except ImportError:
        print("Open3D not installed. Install with: pip install open3d")
        print("Falling back to matplotlib visualization...")
        return False
    
    return True


# Example usage combining with the loader
if __name__ == "__main__":
    from ReadPCD import load_binary_pcd  # Assuming previous code is in load_pcd.py
    
    # Load the PCD file
    points, colors, intensity, variance, offset = load_binary_pcd('annotations/input.pcd')
    
    print(f"\nVisualizing {len(points):,} points")
    
    # Try Open3D first (best interactive experience)
    success = visualize_with_open3d(points, colors)
    
    # If Open3D not available, use matplotlib
    if not success:
        # Subsample for performance if too many points
        subsample_size = 50000 if len(points) > 50000 else None
        visualize_point_cloud(points, colors, intensity, subsample=subsample_size)
        
        # Show intensity histogram
        visualize_intensity_histogram(intensity)