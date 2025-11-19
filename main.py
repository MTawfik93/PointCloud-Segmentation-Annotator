# main.py
import json
import os
from Viewer import PointCloudViewer

def load_config(config_path: str = "config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    file_name = config.get('file_name')
    folder_name = config.get('folder_name')
    
    if file_name and folder_name:
        raise ValueError("Specify either 'file_name' or 'folder_name', not both.")
    if not file_name and not folder_name:
        raise ValueError("Must specify either 'file_name' or 'folder_name'.")
    
    required = ['save_location', 'num_classes', 'class_names', 'class_colors', 'undo_steps', 'unbrush_class']
    for key in required:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")
    
    os.makedirs(config['save_location'], exist_ok=True)
    
    return config

def main():
    config = load_config()
    
    save_location = config['save_location']
    class_names   = config['class_names']
    class_colors  = [[c/255.0 for c in rgb] for rgb in config['class_colors']]
    undo_steps    = config['undo_steps']
    unbrush_class = config['unbrush_class']
    pcd_folder    = config["folder_name"]

    print(f"Save location: {save_location}")
    print(f"Classes: {class_names}")
    print(f"Undo steps: {undo_steps}")
    
    file_name = config.get('file_name')
    #folder_name = config.get('folder_name')
    
    # Get list of PCD files
    if file_name:
        pcd_paths = [os.path.join(pcd_folder or '', file_name)]
    else:
        pcd_paths = sorted([
            os.path.join(pcd_folder, f) for f in os.listdir(pcd_folder)
            if f.lower().endswith('.pcd')
        ])
        if not pcd_paths:
            raise FileNotFoundError(f"No .pcd files in {pcd_folder}")
    
    print(f"Found {len(pcd_paths)} file(s)")

    viewer = PointCloudViewer(
        save_location=save_location,
        class_names=class_names,
        class_colors=class_colors,
        undo_steps=undo_steps,
        unbrush_class=unbrush_class,
        pcd_folder=pcd_folder,
        config=config
    )
    viewer.run()

if __name__ == "__main__":
    main()