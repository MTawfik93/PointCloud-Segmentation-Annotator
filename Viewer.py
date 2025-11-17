# Viewer.py
import glfw
import zlib
import lzf
import struct
import open3d as o3d
from OpenGL.GL import *
import numpy as np
from pyrr import Matrix44, matrix44
import imgui
from imgui.integrations.glfw import GlfwRenderer
from utilities import apply_viridis
import json
import os

def try_import_open3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        raise ImportError("open3d not installed → pip install open3d")

class Trackball:
    def __init__(self, eye, center, up):
        self.eye = np.array(eye, dtype=np.float32)
        self.center = np.array(center, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.last_pos = None
        self.zoom = 1.0

    def mouse_down(self, x, y): self.last_pos = (x, y)
    def mouse_up(self): self.last_pos = None

    def mouse_move(self, x, y, w, h):
        if not self.last_pos: return
        dx = (x - self.last_pos[0]) / w
        dy = (y - self.last_pos[1]) / h
        angle = np.linalg.norm([dx, dy]) * 8
        if angle == 0: return
        axis = np.array([-dy, dx, 0.0])
        rot = matrix44.create_from_axis_rotation(axis, angle)
        dir_vec = self.eye - self.center
        self.eye = self.center + matrix44.apply_to_vector(rot, dir_vec)
        self.last_pos = (x, y)

    def zoom_in(self): self.zoom *= 0.9
    def zoom_out(self): self.zoom *= 1.1

    def get_view(self):
        dist = np.linalg.norm(self.eye - self.center) * self.zoom
        eye = self.center + (self.eye - self.center) / np.linalg.norm(self.eye - self.center) * dist
        return Matrix44.look_at(eye, self.center, self.up)

class LOD:
    def __init__(self, pts, cols, max_pts=5_000_000):
        self.full_pts = pts
        self.full_col = cols
        self.max_pts = max_pts
        self.current_pts = pts
        self.current_col = cols

    def update(self, scale):
        n = len(self.full_pts)
        if n <= self.max_pts or scale >= 1.0:
            self.current_pts = self.full_pts
            self.current_col = self.full_col
            return
        step = max(1, int(1 / scale))
        idx = np.arange(0, n, step)[:self.max_pts]
        self.current_pts = self.full_pts[idx]
        self.current_col = self.full_col[idx]

class PointCloudViewer:
    def __init__(self, save_location, class_names, class_colors, undo_steps, unbrush_class, pcd_paths, config):
        self.win = None
        self.width = 1400
        self.height = 800
        self.prog = None
        self.vao = None
        self.vbo_pos = None
        self.vbo_col = None
        self.lod = None
        self.trackball = None
        self.proj = None
        self.translation = np.zeros(3, dtype=np.float32)
        self.panning = False
        self.last_pan = None
        self.imgui_io = None
        self.imgui_renderer = None

        # Config
        self.save_location = save_location
        self.class_names = class_names
        self.class_colors = np.array(class_colors, dtype=np.float32)
        self.undo_steps = undo_steps
        self.unbrush_class = unbrush_class
        self.pcd_paths = pcd_paths
        self.current_file_idx = 0

        # Annotation
        self.brush_active = False
        self.unbrush_mode = False
        self.brush_radius = 0.5
        self.selected_class = 0
        self.annotations = {}  # idx → class_id
        self.original_colors = None
        self.is_painting = False

        # Undo/Redo
        self.undo_stack = []  # list of (indices, old_class_dict)
        self.redo_stack = []

        # Filteration
        self.outlier_threshold = config.get("outlier_threshold", 99998.0)
        self.filter_on_load = config.get("filter_on_load", False)
        self.filtered_once = False

    def load_binary_compressed_pcd(self,filepath):
        """
        Load a binary compressed PCD file with LZF compression.
        
        Returns:
            points: Nx3 array of xyz coordinates (float32)
            colors: Nx3 array of RGB colors (0-255 uint8)
        """
        with open(filepath, 'rb') as f:
            # Parse header
            header = {}
            while True:
                line = f.readline().decode('ascii').strip()
                if line.startswith('DATA'):
                    data_type = line.split()[1]
                    header['DATA'] = data_type
                    break
                if line:
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        key, value = parts
                        header[key] = value
            
            # Read compressed binary data
            compressed_data = f.read()
        
        num_points = int(header.get('POINTS', 0))
        print(f"Expected points: {num_points}")
        
        # Decompress data
        compressed_size = struct.unpack('<I', compressed_data[0:4])[0]
        uncompressed_size = struct.unpack('<I', compressed_data[4:8])[0]
        
        print(f"Compressed: {compressed_size} bytes → Uncompressed: {uncompressed_size} bytes")
        
        try:
            
            binary_data = lzf.decompress(compressed_data[8:8+compressed_size], uncompressed_size)
            print("Decompressed successfully with LZF")
        except ImportError:
            raise ImportError("python-lzf not installed. Install with: pip install python-lzf")
        except Exception as e:
            raise RuntimeError(f"Decompression error: {e}")
        
        # Calculate field sizes (x, y, z, variance, intensity, offset)
        field_sizes = {
            'x': num_points * 4,
            'y': num_points * 4,
            'z': num_points * 4,
            'variance': num_points * 4,
            'intensity': num_points * 2,
            'offset': num_points * 2
        }
        
        # Parse field-by-field
        offset_pos = 0
        x_data = np.frombuffer(binary_data[offset_pos:offset_pos+field_sizes['x']], dtype='<f4')
        offset_pos += field_sizes['x']
        
        y_data = np.frombuffer(binary_data[offset_pos:offset_pos+field_sizes['y']], dtype='<f4')
        offset_pos += field_sizes['y']
        
        z_data = np.frombuffer(binary_data[offset_pos:offset_pos+field_sizes['z']], dtype='<f4')
        offset_pos += field_sizes['z']
        
        variance = np.frombuffer(binary_data[offset_pos:offset_pos+field_sizes['variance']], dtype='<f4')
        offset_pos += field_sizes['variance']
        
        intensity = np.frombuffer(binary_data[offset_pos:offset_pos+field_sizes['intensity']], dtype='<u2')

        offset_data = np.frombuffer(binary_data[offset_pos:offset_pos+field_sizes['offset']], dtype='<u2')
        
        # Stack into Nx3 array
        points = np.stack([x_data, y_data, z_data], axis=1)
        
        # Remove NaN and outlier points
        valid_mask = ~np.isnan(points).any(axis=1) & ~np.isnan(intensity)
        outlier_mask = (np.abs(points[:, 0]) <= 99999) & \
                    (np.abs(points[:, 1]) <= 99999) & \
                    (np.abs(points[:, 2]) <= 99999)
        
        combined_mask = valid_mask & outlier_mask
        points = points[combined_mask]
        intensity = intensity[combined_mask]
        
        print(f"After filtering: {len(points)} points remain")
        # print(f"\nIntensity statistics:")
        # print(f"Min: {intensity.min()}, Max: {intensity.max()}")
        # print(f"Mean: {intensity.mean():.2f}, Std: {intensity.std():.2f}")
        # print(f"Data type: {intensity.dtype}")
        # print(f"Sample values: {intensity[:10]}")
        # Convert intensity to grayscale colors (0-255 range)
        if intensity.max() > 0:
            print("Converting intensity to grayscale colors")
            colors = (intensity.astype(np.float64) / intensity.max() * 255.0).astype(np.uint8)
        else:
            colors = intensity.astype(np.uint8) / 255.0
        
        colors = np.stack([colors, colors, colors], axis=1)
        
        return points, colors

    def load_standard_pcd(self, path: str):
        """
        Load a standard PCD file using Open3D.
        
        Returns:
            points: Nx3 array of xyz coordinates (float32)
            colors: Nx3 array of RGB colors (0-1 float32 range)
        """
        try:
            o3d = try_import_open3d()
            pc = o3d.io.read_point_cloud(path)
            
            if pc.is_empty():
                raise ValueError("PCD file is empty")
            
            points = np.asarray(pc.points, dtype=np.float32)
            
            if pc.has_colors():
                colors = np.asarray(pc.colors, dtype=np.float32)
                print(f"Open3D loaded {len(points)} points with colors")
            else:
                # Fallback: medium gray
                colors = np.ones((len(points), 3), dtype=np.float32) * 0.7
                print(f"Open3D loaded {len(points)} points (no color data, using gray)")
            
            return points, colors
            
        except Exception as e:
            raise RuntimeError(f"Open3D failed to load PCD: {e}")
    def load_pointcloud(self, path: str):
        """
        Load a PCD file, automatically handling binary_compressed or standard formats.
        
        Args:
            path: Path to the PCD file
            
        Returns:
            points: Nx3 array of xyz coordinates
            colors: Nx3 array of RGB colors (0-255 range for binary_compressed, 0-1 range for standard)
        """
        # First, check if it's binary_compressed by reading the header
        with open(path, 'rb') as f:
            header = {}
            while True:
                line = f.readline().decode('ascii').strip()
                if line.startswith('DATA'):
                    data_type = line.split()[1]
                    header['DATA'] = data_type
                    break
                if line:
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        key, value = parts
                        header[key] = value
        
        # Route to appropriate loader
        if header.get('DATA') == 'binary_compressed':
            print("Detected binary_compressed format, using custom loader...")
            return self.load_binary_compressed_pcd(path)
        else:
            print("Detected standard PCD format, using Open3D loader...")
            return self.load_standard_pcd(path)

    def init_gl(self):
        if not glfw.init(): raise RuntimeError("GLFW init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, GL_FALSE)  # NO RESIZE
        self.win = glfw.create_window(1400, 800, "Point Cloud Annotator", None, None)
        glfw.make_context_current(self.win)

        imgui.create_context()
        self.imgui_io = imgui.get_io()
        self.imgui_io.ini_file_name = b""
        self.imgui_renderer = GlfwRenderer(self.win)

        vs = """
        #version 330 core
        layout(location=0) in vec3 pos;
        layout(location=1) in vec3 col;
        uniform mat4 mvp;
        out vec3 vcol;
        void main(){
            gl_Position = mvp * vec4(pos,1.0);
            gl_PointSize = 6.0;
            vcol = col;
        }
        """
        fs = """
        #version 330 core
        in vec3 vcol;
        out vec4 frag;
        void main(){
            vec2 c = gl_PointCoord - 0.5;
            if(dot(c,c)>0.25) discard;
            frag = vec4(vcol,1.0);
        }
        """
        def compile(src, typ):
            s = glCreateShader(typ)
            glShaderSource(s, src)
            glCompileShader(s)
            if not glGetShaderiv(s, GL_COMPILE_STATUS):
                raise RuntimeError(glGetShaderInfoLog(s).decode())
            return s
        self.prog = glCreateProgram()
        glAttachShader(self.prog, compile(vs, GL_VERTEX_SHADER))
        glAttachShader(self.prog, compile(fs, GL_FRAGMENT_SHADER))
        glLinkProgram(self.prog)

        self.vao = glGenVertexArrays(1)
        self.vbo_pos, self.vbo_col = glGenBuffers(2)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

        glUseProgram(self.prog)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.05, 0.05, 0.1, 1)

    def get_ray(self, mx, my):
        # Get current window size
        width, height = glfw.get_window_size(self.win)
        if width == 0 or height == 0:
            return np.zeros(3), np.array([0, 0, 1])

        # NDC from screen coordinates
        ndc_x = (2.0 * mx) / width - 1.0
        ndc_y = 1.0 - (2.0 * my) / height

        # Clip space
        clip_near = np.array([ndc_x, ndc_y, -1.0, 1.0])
        clip_far  = np.array([ndc_x, ndc_y,  1.0, 1.0])

        # Full MVP inverse = inverse(proj * view * model)
        model = Matrix44.from_translation(self.translation)
        view = self.trackball.get_view()
        proj = self.proj

        mvp = proj * view * model
        inv_mvp = mvp.inverse

        # Unproject near and far
        near_h = matrix44.apply_to_vector(inv_mvp, clip_near[:3])
        far_h  = matrix44.apply_to_vector(inv_mvp, clip_far[:3])

        # Ray in world space
        direction = far_h - near_h
        direction /= np.linalg.norm(direction) + 1e-8
        return near_h, direction

    def pick_points_under_cursor(self, mx, my):
        if mx <= 200: return np.array([])
        eye, direction = self.get_ray(mx, my)
        vecs = self.lod.full_pts - eye
        proj = np.dot(vecs, direction)
        lateral = np.cross(vecs, direction)
        lateral_dist = np.linalg.norm(lateral, axis=1)
        mask = (proj > 0) & (lateral_dist < self.brush_radius)
        return np.where(mask)[0]

    def update_colors(self):
        colors = self.original_colors.copy()
        for idx, cls in self.annotations.items():
            colors[idx] = self.class_colors[cls]
        self.lod.current_col = colors

    def save_annotations(self):
        base = os.path.basename(self.pcd_paths[self.current_file_idx])
        name = os.path.splitext(base)[0]
        ann_path = os.path.join(self.save_location, f"{name}.json")
        data = {
            "pcd": self.pcd_paths[self.current_file_idx],
            "annotations": {str(i): int(c) for i, c in self.annotations.items()}
        }
        with open(ann_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {ann_path}")

    def export_labeled_pcd(self):
        base = os.path.basename(self.pcd_paths[self.current_file_idx])
        name = os.path.splitext(base)[0]
        out_path = os.path.join(self.save_location, f"{name}_labeled.pcd")
        o3d = try_import_open3d()
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.lod.full_pts)
        colors = self.original_colors.copy()
        for idx, cls in self.annotations.items():
            colors[idx] = self.class_colors[cls]
        pc.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(out_path, pc)
        print(f"Exported: {out_path}")

    def push_undo(self, changed_indices):
        if len(self.undo_stack) >= self.undo_steps:
            self.undo_stack.pop(0)
        old_state = {}
        for idx in changed_indices:
            old_state[idx] = self.annotations.get(idx, -1)  # -1 = not labeled
        self.undo_stack.append((changed_indices.copy(), old_state))
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack: return
        indices, old_state = self.undo_stack.pop()
        # Save current for redo
        current_state = {idx: self.annotations.get(idx, -1) for idx in indices}
        self.redo_stack.append((indices.copy(), current_state))
        # Restore old
        for idx, old_cls in old_state.items():
            if old_cls == -1:
                self.annotations.pop(idx, None)
            else:
                self.annotations[idx] = old_cls
        self.update_colors()
        self.upload_colors()

    def redo(self):
        if not self.redo_stack: return
        indices, old_state = self.redo_stack.pop()
        current_state = {idx: self.annotations.get(idx, -1) for idx in indices}
        self.undo_stack.append((indices.copy(), current_state))
        for idx, old_cls in old_state.items():
            if old_cls == -1:
                self.annotations.pop(idx, None)
            else:
                self.annotations[idx] = old_cls
        self.update_colors()
        self.upload_colors()

    def upload_colors(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
        glBufferData(GL_ARRAY_BUFFER, self.lod.current_col.nbytes, self.lod.current_col, GL_STATIC_DRAW)

    def load_file(self, idx):
        self.current_file_idx = np.clip(idx, 0, len(self.pcd_paths)-1)
        path = self.pcd_paths[self.current_file_idx]
        print(f"Loading: {path}")
        pts, cols = self.load_pointcloud(path)
        
        # Track if this is binary compressed format
        is_binary_compressed = cols.dtype == np.uint8
        
        # Normalize colors to 0-1 range if needed
        if is_binary_compressed:
            # Binary compressed format returns 0-255 uint8
            cols = cols.astype(np.float32) / 255.0
            print("Binary compressed: using intensity as grayscale")
        else:
            # Standard PCD: check if colors are grayscale (intensity data)
            is_intensity = cols.shape[1] == 3 and np.allclose(cols[:,0], cols[:,1]) and np.allclose(cols[:,1], cols[:,2])
            
            if is_intensity:
                # Apply viridis colormap to intensity values
                cols = apply_viridis(cols[:, 0])
                print("Applied viridis colormap to intensity data")
        
        self.lod = LOD(pts, cols)
        self.original_colors = cols.copy()
        center = pts.mean(axis=0)
        radius = np.linalg.norm(pts - center, axis=1).max()
        self.lod_radius = radius
        eye = center + np.array([1.2, 0.8, 1.2]) * radius * 3
        self.trackball = Trackball(eye, center, [0, 1, 0])
        width, height = glfw.get_window_size(self.win)
        self.proj = Matrix44.perspective_projection(45, width/height, 0.01, radius * 20)
        self.update_projection()
        
        # Auto-filter on load?
        if self.filter_on_load and not self.filtered_once:
            self.filter_outliers()
        else:
            self.filtered_once = False
        
        self.annotations = {}
        self.undo_stack = []
        self.redo_stack = []
        self.update_colors()
        self.upload_colors()
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, self.lod.current_pts.nbytes, self.lod.current_pts, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
        glBufferData(GL_ARRAY_BUFFER, self.lod.current_col.nbytes, self.lod.current_col, GL_STATIC_DRAW)
    def update_projection(self):
        width, height = glfw.get_window_size(self.win)
        if height == 0: height = 1
        self.proj = Matrix44.perspective_projection(45, width/height, 0.01, self.lod_radius * 20)

    def filter_outliers(self):
        if self.filtered_once:
            print("Already filtered!")
            return

        pts = self.lod.full_pts
        valid = np.ones(len(pts), dtype=bool)

        # Remove NaN / inf
        valid &= np.isfinite(pts).all(axis=1)

        # Remove points where ANY x, y, or z > threshold
        valid &= np.abs(pts).max(axis=1) < self.outlier_threshold

        if not np.all(valid):
            old_count = len(pts)
            self.lod.full_pts = pts[valid]
            self.lod.full_col = self.lod.current_col[valid]
            self.original_colors = self.original_colors[valid]

            # Rebuild LOD
            self.lod.current_pts = self.lod.full_pts
            self.lod.current_col = self.lod.full_col

            new_count = len(self.lod.full_pts)
            removed = old_count - new_count
            print(f"Filtered outliers: {old_count} → {new_count} points (-{removed})")

            # Re-center view
            center = self.lod.full_pts.mean(axis=0)
            radius = np.linalg.norm(self.lod.full_pts - center, axis=1).max()
            self.lod_radius = radius
            self.trackball.center = center
            self.trackball.eye = center + np.array([1.2, 0.8, 1.2]) * radius * 3

            # Update GPU
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
            glBufferData(GL_ARRAY_BUFFER, self.lod.full_pts.nbytes, self.lod.full_pts, GL_STATIC_DRAW)
            self.upload_colors()

        self.filtered_once = True

    def run(self):
        self.init_gl()
        self.load_file(0)

        def mouse_button(window, button, action, mods):
            if imgui.get_io().want_capture_mouse: return
            x, y = glfw.get_cursor_pos(window)

            if button == glfw.MOUSE_BUTTON_LEFT:
                if action == glfw.PRESS:
                    if self.brush_active and x > 200:
                        self.is_painting = True
                        indices = self.pick_points_under_cursor(x, y)
                        if indices.size > 0:
                            self.push_undo(indices)
                            if self.unbrush_mode:
                                for idx in indices:
                                    self.annotations.pop(idx, None)
                            else:
                                for idx in indices:
                                    self.annotations[idx] = self.selected_class
                            self.update_colors()
                            self.upload_colors()
                            # UPLOAD TO GPU
                            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
                            glBufferData(GL_ARRAY_BUFFER, self.lod.current_col.nbytes, self.lod.current_col, GL_STATIC_DRAW)
                    else:
                        self.trackball.mouse_down(x, y)  # orbit
                else:
                    self.is_painting = False
                    self.trackball.mouse_up()  # orbit release

            elif button == glfw.MOUSE_BUTTON_RIGHT and (mods & glfw.MOD_SHIFT):
                if action == glfw.PRESS:
                    self.panning = True
                    self.last_pan = (x, y)
                else:
                    self.panning = False  # FIXED: stop on release
                    self.last_pan = None

        def cursor_pos(window, xpos, ypos):
            if imgui.get_io().want_capture_mouse: return

            # --- PANNING ---
            if self.panning and self.last_pan:
                dx = xpos - self.last_pan[0]
                dy = ypos - self.last_pan[1]
                view_dir = self.trackball.eye - self.trackball.center
                view_dir /= np.linalg.norm(view_dir) + 1e-8
                world_up = np.array([0,1,0])
                right = np.cross(view_dir, world_up)
                right /= np.linalg.norm(right) + 1e-8
                up = np.cross(right, view_dir)
                dist = np.linalg.norm(self.trackball.eye - self.trackball.center) * self.trackball.zoom
                scale = np.tan(np.radians(45)/2) * dist * 2 / self.height
                self.translation -= right * (dx * scale) + up * (dy * scale)
                self.last_pan = (xpos, ypos)

            elif self.is_painting and self.brush_active and xpos > 200:
                indices = self.pick_points_under_cursor(xpos, ypos)
                if indices.size > 0:
                    self.push_undo(indices)
                    if self.unbrush_mode:
                        for idx in indices:
                            self.annotations.pop(idx, None)
                    else:
                        for idx in indices:
                            self.annotations[idx] = self.selected_class
                    self.update_colors()
                    self.upload_colors()
                    # UPLOAD TO GPU
                    glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
                    glBufferData(GL_ARRAY_BUFFER, self.lod.current_col.nbytes, self.lod.current_col, GL_STATIC_DRAW)

            # --- ORBIT ROTATION ---
            elif self.trackball.last_pos and not self.brush_active:
                self.trackball.mouse_move(xpos, ypos, self.width, self.height)

        def scroll(window, xoff, yoff):
            if imgui.get_io().want_capture_mouse: return

            if self.brush_active:
                self.brush_radius = np.clip(self.brush_radius + yoff * 0.1, 0.1, 5.0)
            else:
                if yoff > 0:
                    self.trackball.zoom_in()
                else:
                    self.trackball.zoom_out()
                self.update_projection()  # CRITICAL: update proj on zoom


        def key_callback(window, key, scancode, action, mods):
            if action == glfw.PRESS:
                if key == glfw.KEY_Z and mods & glfw.MOD_CONTROL:
                    self.undo()
                elif key == glfw.KEY_Y and mods & glfw.MOD_CONTROL:
                    self.redo()
                elif key == glfw.KEY_LEFT:
                    self.load_file(self.current_file_idx - 1)
                elif key == glfw.KEY_RIGHT:
                    self.load_file(self.current_file_idx + 1)

        glfw.set_mouse_button_callback(self.win, mouse_button)
        glfw.set_cursor_pos_callback(self.win, cursor_pos)
        glfw.set_scroll_callback(self.win, scroll)
        glfw.set_key_callback(self.win, key_callback)

        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            self.imgui_renderer.process_inputs()
            imgui.new_frame()

            # --- Sidebar ---
            imgui.set_next_window_position(0, 0)
            imgui.set_next_window_size(200, self.height)
            imgui.begin("Tools", True,
                        imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR)

            imgui.text(f"File: {self.current_file_idx+1}/{len(self.pcd_paths)}")
            imgui.separator()

            changed, self.brush_active = imgui.checkbox("Brush Tool", self.brush_active)
            if changed and self.brush_active:
                self.unbrush_mode = False

            if self.brush_active:
                _, self.unbrush_mode = imgui.checkbox("Unbrush Mode", self.unbrush_mode)

            imgui.text(f"Radius: {self.brush_radius:.2f} m")

            imgui.push_item_width(150)
            _, self.selected_class = imgui.combo("Class", self.selected_class, self.class_names)
            imgui.pop_item_width()

            if imgui.button("Undo (Ctrl+Z)", width=180):
                self.undo()
            if imgui.button("Redo (Ctrl+Y)", width=180):
                self.redo()

            if imgui.button("Save JSON", width=180):
                self.save_annotations()
            if imgui.button("Export PCD", width=180):
                self.export_labeled_pcd()
            if imgui.button("Filter Outliers", width=180):
                self.filter_outliers()
                # if self.filtered_once:
                #     imgui.text_colored((0.0, 1.0, 0.0, 1.0), "Outliers filtered!")
            imgui.text(f"Labeled: {len(self.annotations)}")

            imgui.end()

            # --- 3D Input ---
            if self.trackball.last_pos and not imgui.get_io().want_capture_mouse and not self.brush_active:
                x, y = glfw.get_cursor_pos(self.win)
                self.trackball.mouse_move(x, y, self.width, self.height)

            # LOD
            if self.lod:
                dist = np.linalg.norm(self.trackball.eye - self.trackball.center) * self.trackball.zoom
                scale = min(1.0, self.lod_radius * 3 / dist)
                self.lod.update(scale)
                if self.lod.current_pts.shape[0] * 12 != glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE):
                    glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
                    glBufferData(GL_ARRAY_BUFFER, self.lod.current_pts.nbytes, self.lod.current_pts, GL_STATIC_DRAW)

            model = Matrix44.from_translation(self.translation)
            view = self.trackball.get_view()
            mvp = self.proj * view * model
            glUniformMatrix4fv(glGetUniformLocation(self.prog, "mvp"), 1, GL_FALSE, mvp)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glBindVertexArray(self.vao)
            glDrawArrays(GL_POINTS, 0, self.lod.current_pts.shape[0])
            glBindVertexArray(0)

            # --- Brush preview ---
            if self.brush_active:
                mx, my = glfw.get_cursor_pos(self.win)
                if mx > 200:
                    draw_list = imgui.get_background_draw_list()
                    color = imgui.get_color_u32_rgba(1, 1, 1, 0.7) if self.unbrush_mode else imgui.get_color_u32_rgba(*self.class_colors[self.selected_class], 0.7)
                    draw_list.add_circle(mx, my, self.brush_radius * 60, color, thickness=2)

            imgui.render()
            self.imgui_renderer.render(imgui.get_draw_data())
            glfw.swap_buffers(self.win)

        self.imgui_renderer.shutdown()
        glfw.terminate()