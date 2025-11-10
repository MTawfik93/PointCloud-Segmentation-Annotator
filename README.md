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

# Installation

```bash
git clone https://github.com/yourusername/PointCloudAnnotator.git
cd PointCloudAnnotator
pip install glfw imgui[glfw] numpy pyrr open3d PyGLM
```

## Install dependencies
```
pip install glfw imgui[glfw] numpy pyrr open3d PyGLM
```
## Put your point clouds in
```
pointclouds/your_file.pcd
```
## Run
```
python main.py
```
## Controls
```
- Left-click + drag → Paint / Unpaint (when Brush Tool enabled)
- Mouse wheel → Change brush radius
- Left-click + drag (Brush OFF) → Orbit
- Shift + Right-click + drag → Pan
- Left/Right Arrow → Next / Previous file
- Ctrl+Z / Ctrl+Y → Undo / Redo
- Save JSON / Export PCD → Buttons in sidebar
```
## Project Structure
```bash
text├── main.py              # Entry point
├── Viewer.py            # Full OpenGL + ImGui viewer
├── config.json          # Classes, colors, undo steps
├── pointclouds/         # ← put your .pcd files here
├── annotations/         # ← auto-saved .json labels
├── utilities.py         # Viridis colormap helper
└── .gitignore           # Keeps repo clean
```

## Config Example (config.json)
```
json{
  "folder_name": "pointclouds",
  "save_location": "annotations",
  "num_classes": 5,
  "class_names": ["Defect", "Long Defect", "Hole", "Scratch", "Bump"],
  "class_colors": [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[180,180,180]],
  "undo_steps": 10,
  "unbrush_class": 4
}
```
License
MIT © 2025

Happy annotating!
Made with ❤️
