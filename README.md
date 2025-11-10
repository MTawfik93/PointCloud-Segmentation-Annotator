# PointCloudAnnotator

**Fast OpenGL-based 3D point cloud annotation tool** for LiDAR datasets.  
Perfect for defect detection, quality control, and semantic segmentation labeling.

![screenshot](https://via.placeholder.com/800x500.png?text=PointCloudAnnotator+Screenshot)  
*(Replace with your own screenshot)*

## Features
- **Real-time brush painting** with adjustable radius (mouse wheel)
- **Unbrush mode** – restores original colors
- **Undo/Redo** (Ctrl+Z / Ctrl+Y) – up to 10 steps
- **Perfect brush centering** – no drift when zooming, orbiting or panning
- **Instant GPU visualization** – millions of points at 60 FPS
- Auto-load & auto-save annotations (`.json`)
- Export labeled point clouds (`.pcd` with class colors)
- Fixed 1400×800 window – stable brush zero
- Supports `.pcd`, `.ply`, intensity maps, and more

## Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/PointCloudAnnotator.git
cd PointCloudAnnotator

# 2. Install dependencies
pip install glfw imgui[glfw] numpy pyrr open3d PyGLM

# 3. Put your point clouds in
pointclouds/your_file.pcd

# 4. Run
python main.py
