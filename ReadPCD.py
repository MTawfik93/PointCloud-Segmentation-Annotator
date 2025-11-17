import numpy as np
import struct
import lzf

# def load_binary_pcd(filepath):
#     """
#     Load a binary compressed PCD file with specific format.
    
#     PCD Format:
#     - Header with metadata (VERSION, FIELDS, SIZE, TYPE, etc.)
#     - DATA binary_compressed
#     - Compressed binary data (LZF compression)
    
#     Point Format: x y z variance intensity offset
#     Bytes:        4 4 4 4        2         2
    
#     Args:
#         filepath: Path to the PCD file
        
#     Returns:
#         points: Nx3 array of xyz coordinates
#         colors: Nx3 array of RGB colors derived from intensity (0-255)
#         intensity: N array of intensity values
#         variance: N array of variance values
#         offset: N array of offset values
#     """
#     with open(filepath, 'rb') as f:
#         # Parse header
#         header = {}
#         data_start = 0
        
#         while True:
#             line = f.readline().decode('ascii').strip()
            
#             if line.startswith('DATA'):
#                 data_type = line.split()[1]
#                 header['DATA'] = data_type
#                 data_start = f.tell()
#                 break
            
#             if line:
#                 parts = line.split(None, 1)
#                 if len(parts) == 2:
#                     key, value = parts
#                     header[key] = value
        
#         print("PCD Header:")
#         for key, value in header.items():
#             print(f"  {key}: {value}")
        
#         # Read compressed binary data
#         compressed_data = f.read()
#         print(f"\nCompressed data size: {len(compressed_data)} bytes")
    
#     # Get expected number of points
#     num_points = int(header.get('POINTS', 0))
#     print(f"Expected points: {num_points}")
    
#     # Decompress data
#     if header['DATA'] == 'binary_compressed':
#         print("Decompressing binary_compressed data...")
        
#         # PCD binary_compressed format:
#         # 4 bytes: compressed size
#         # 4 bytes: uncompressed size
#         # remaining: LZF compressed data
        
#         compressed_size = struct.unpack('<I', compressed_data[0:4])[0]
#         uncompressed_size = struct.unpack('<I', compressed_data[4:8])[0]
        
#         print(f"Compressed size: {compressed_size} bytes")
#         print(f"Uncompressed size: {uncompressed_size} bytes")
        
#         # Try to decompress with LZF
#         try:
#             import lzf
#             binary_data = lzf.decompress(compressed_data[8:8+compressed_size], uncompressed_size)
#             print("Decompressed successfully with LZF")
#         except ImportError:
#             print("Warning: python-lzf not installed. Trying alternative decompression...")
#             try:
#                 import zlib
#                 # Try zlib decompression as fallback
#                 binary_data = zlib.decompress(compressed_data)
#                 print("Decompressed successfully with zlib")
#             except:
#                 print("Error: Could not decompress data. Install python-lzf: pip install python-lzf")
#                 return None, None, None, None, None
#         except Exception as e:
#             print(f"Decompression error: {e}")
#             return None, None, None, None, None
#     else:
#         # Uncompressed binary
#         binary_data = compressed_data
    
#     print(f"Binary data size after decompression: {len(binary_data)} bytes")
    
#     # Each point: 4 + 4 + 4 + 4 + 2 + 2 = 20 bytes
#     point_size = 20
#     actual_points = len(binary_data) // point_size
    
#     print(f"Actual points in data: {actual_points}")
    
#     # Initialize arrays
#     points = np.zeros((actual_points, 3), dtype=np.float32)
#     variance = np.zeros(actual_points, dtype=np.float32)
#     intensity = np.zeros(actual_points, dtype=np.float32)
#     offset = np.zeros(actual_points, dtype=np.float32)
    
#     # Parse binary data
#     print("Parsing point data...")
#     for i in range(actual_points):
#         start_idx = i * point_size
#         chunk = binary_data[start_idx:start_idx + point_size]
        
#         # Unpack: 4 floats (x,y,z,variance) + 2 unsigned shorts (intensity, offset)
#         # '<' means little-endian
#         x, y, z, var = struct.unpack('<ffff', chunk[0:16])
#         intens, off = struct.unpack('<HH', chunk[16:20])
#         if (i %10000) == 0 and i > 0:
#             print("Point %d is (%f, %f, %f)" % (i, x, y, z))
#         points[i] = [x, y, z]
#         variance[i] = var
#         intensity[i] = intens
#         offset[i] = off
    
#     print("Point data parsed successfully")
    
#     # 1. Remove NaN points
#     valid_mask = ~np.isnan(points).any(axis=1) & ~np.isnan(variance)
#     points = points[valid_mask]
#     variance = variance[valid_mask]
#     intensity = intensity[valid_mask]
#     offset = offset[valid_mask]
    
#     print(f"After removing NaN: {len(points)} points remain")
    
#     # 2. Remove outliers (absolute value > 99999)
#     outlier_mask = (np.abs(points[:, 0]) <= 99999) & \
#                    (np.abs(points[:, 1]) <= 99999) & \
#                    (np.abs(points[:, 2]) <= 99999)
#     points = points[outlier_mask]
#     variance = variance[outlier_mask]
#     intensity = intensity[outlier_mask]
#     offset = offset[outlier_mask]
    
#     print(f"After removing outliers (>99999): {len(points)} points remain")
    
#     # 3. Remove statistical outliers using IQR method
#     # inlier_mask = np.ones(len(points), dtype=bool)
    
#     # for dim in range(3):
#     #     q1 = np.percentile(points[:, dim], 25)
#     #     q3 = np.percentile(points[:, dim], 75)
#     #     iqr = q3 - q1
#     #     lower_bound = q1 - 1.5 * iqr
#     #     upper_bound = q3 + 1.5 * iqr
#     #     inlier_mask &= (points[:, dim] >= lower_bound) & (points[:, dim] <= upper_bound)
    
#     # points = points[inlier_mask]
#     # variance = variance[inlier_mask]
#     # intensity = intensity[inlier_mask]
#     # offset = offset[inlier_mask]
    
#     # print(f"After removing statistical outliers: {len(points)} points remain")
    
#     # # 4. Remove outliers in variance
#     # if len(variance) > 0 and not np.all(np.isnan(variance)):
#     #     var_q1 = np.percentile(variance, 25)
#     #     var_q3 = np.percentile(variance, 75)
#     #     var_iqr = var_q3 - var_q1
#     #     var_lower = var_q1 - 1.5 * var_iqr
#     #     var_upper = var_q3 + 1.5 * var_iqr
#     #     var_mask = (variance >= var_lower) & (variance <= var_upper)
        
#     #     points = points[var_mask]
#     #     variance = variance[var_mask]
#     #     intensity = intensity[var_mask]
#     #     offset = offset[var_mask]
        
#     #     print(f"After removing variance outliers: {len(points)} points remain")
    
#     # 5. Convert intensity to grayscale colors (0-255 range)
#     colors = (intensity.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
#     # Stack to create Nx3 array for RGB (all channels same for grayscale)
#     colors = np.stack([colors, colors, colors], axis=1)
    
#     return points, colors, intensity, variance, offset

def load_binary_pcd(filepath):
    """
    Load a binary compressed PCD file with specific format.
    
    PCD Format:
    - Header with metadata (VERSION, FIELDS, SIZE, TYPE, etc.)
    - DATA binary_compressed
    - Compressed binary data (LZF compression)
    
    Point Format: x y z variance intensity offset
    Bytes:        4 4 4 4        2         2
    
    Args:
        filepath: Path to the PCD file
        
    Returns:
        points: Nx3 array of xyz coordinates
        colors: Nx3 array of RGB colors derived from intensity (0-255)
        intensity: N array of intensity values
        variance: N array of variance values
        offset: N array of offset values
    """
    with open(filepath, 'rb') as f:
        # Parse header
        header = {}
        data_start = 0
        
        while True:
            line = f.readline().decode('ascii').strip()
            
            if line.startswith('DATA'):
                data_type = line.split()[1]
                header['DATA'] = data_type
                data_start = f.tell()
                break
            
            if line:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    key, value = parts
                    header[key] = value
        
        print("PCD Header:")
        for key, value in header.items():
            print(f"  {key}: {value}")
        
        # Read compressed binary data
        compressed_data = f.read()
        print(f"\nCompressed data size: {len(compressed_data)} bytes")
    
    # Get expected number of points
    num_points = int(header.get('POINTS', 0))
    print(f"Expected points: {num_points}")
    
    # Decompress data
    if header['DATA'] == 'binary_compressed':
        print("Decompressing binary_compressed data...")
        
        # PCD binary_compressed format:
        # 4 bytes: compressed size
        # 4 bytes: uncompressed size
        # remaining: LZF compressed data
        
        compressed_size = struct.unpack('<I', compressed_data[0:4])[0]
        uncompressed_size = struct.unpack('<I', compressed_data[4:8])[0]
        
        print(f"Compressed size: {compressed_size} bytes")
        print(f"Uncompressed size: {uncompressed_size} bytes")
        
        # Try to decompress with LZF
        try:
            import lzf
            binary_data = lzf.decompress(compressed_data[8:8+compressed_size], uncompressed_size)
            print("Decompressed successfully with LZF")
        except ImportError:
            print("Warning: python-lzf not installed. Trying alternative decompression...")
            try:
                import zlib
                # Try zlib decompression as fallback
                binary_data = zlib.decompress(compressed_data)
                print("Decompressed successfully with zlib")
            except:
                print("Error: Could not decompress data. Install python-lzf: pip install python-lzf")
                return None, None, None, None, None
        except Exception as e:
            print(f"Decompression error: {e}")
            return None, None, None, None, None
    else:
        # Uncompressed binary
        binary_data = compressed_data
    
    print(f"Binary data size after decompression: {len(binary_data)} bytes")
    
    # Calculate actual number of points based on field sizes
    # FIELDS: x y z variance intensity offset
    # SIZE:   4 4 4 4        2         2
    # Data is stored field-by-field (all x, then all y, etc.)
    
    field_sizes = {
        'x': num_points * 4,          # float32
        'y': num_points * 4,          # float32
        'z': num_points * 4,          # float32
        'variance': num_points * 4,   # float32
        'intensity': num_points * 2,  # uint16
        'offset': num_points * 2      # uint16
    }
    
    expected_size = sum(field_sizes.values())
    print(f"Expected binary size: {expected_size} bytes")
    print(f"Actual binary size: {len(binary_data)} bytes")
    
    if len(binary_data) < expected_size:
        print(f"Warning: Binary data smaller than expected!")
        # Recalculate num_points based on actual data
        actual_points = len(binary_data) // 20  # fallback to point-by-point if needed
    else:
        actual_points = num_points
    
    # Parse field-by-field (all x values, then all y values, etc.)
    print("Parsing point data field-by-field...")
    
    try:
        offset_pos = 0
        
        # Read x coordinates
        x_data = np.frombuffer(binary_data[offset_pos:offset_pos+field_sizes['x']], dtype='<f4')
        offset_pos += field_sizes['x']
        print(f"Parsed {len(x_data)} x values")
        
        # Read y coordinates
        y_data = np.frombuffer(binary_data[offset_pos:offset_pos+field_sizes['y']], dtype='<f4')
        offset_pos += field_sizes['y']
        print(f"Parsed {len(y_data)} y values")
        
        # Read z coordinates
        z_data = np.frombuffer(binary_data[offset_pos:offset_pos+field_sizes['z']], dtype='<f4')
        offset_pos += field_sizes['z']
        print(f"Parsed {len(z_data)} z values")
        
        # Read variance
        variance = np.frombuffer(binary_data[offset_pos:offset_pos+field_sizes['variance']], dtype='<f4')
        offset_pos += field_sizes['variance']
        print(f"Parsed {len(variance)} variance values")
        
        # Read intensity
        intensity = np.frombuffer(binary_data[offset_pos:offset_pos+field_sizes['intensity']], dtype='<u2')
        offset_pos += field_sizes['intensity']
        print(f"Parsed {len(intensity)} intensity values")
        
        # Read offset
        offset_data = np.frombuffer(binary_data[offset_pos:offset_pos+field_sizes['offset']], dtype='<u2')
        print(f"Parsed {len(offset_data)} offset values")
        
        # Stack into Nx3 array
        points = np.stack([x_data, y_data, z_data], axis=1)
        
        # Print sample points
        print("\nSample points (field-by-field parsing):")
        for i in [0, 10000, 50000, 100000, len(points)//2, len(points)-1]:
            if i < len(points):
                print(f"Point {i}: ({points[i,0]:.6f}, {points[i,1]:.6f}, {points[i,2]:.6f})")
        
        print("Point data parsed successfully")
        
    except Exception as e:
        print(f"Field-by-field parsing failed: {e}")
        print("Falling back to point-by-point parsing...")
        
        # Fallback to original point-by-point parsing
        point_size = 20
        actual_points = len(binary_data) // point_size
        
        points = np.zeros((actual_points, 3), dtype=np.float32)
        variance = np.zeros(actual_points, dtype=np.float32)
        intensity = np.zeros(actual_points, dtype=np.float32)
        offset_data = np.zeros(actual_points, dtype=np.float32)
        
        for i in range(actual_points):
            start_idx = i * point_size
            chunk = binary_data[start_idx:start_idx + point_size]
            
            x, y, z, var = struct.unpack('<ffff', chunk[0:16])
            intens, off = struct.unpack('<HH', chunk[16:20])
            
            if (i % 10000) == 0 and i > 0:
                print(f"Point {i} is ({x:.6f}, {y:.6f}, {z:.6f})")
            
            points[i] = [x, y, z]
            variance[i] = var
            intensity[i] = intens
            offset_data[i] = off
    
    # 1. Remove NaN points
    valid_mask = ~np.isnan(points).any(axis=1) & ~np.isnan(variance)
    points = points[valid_mask]
    variance = variance[valid_mask]
    intensity = intensity[valid_mask]
    offset_data = offset_data[valid_mask]
    
    print(f"After removing NaN: {len(points)} points remain")
    
    # 2. Remove outliers (absolute value > 99999)
    outlier_mask = (np.abs(points[:, 0]) <= 99999) & \
                   (np.abs(points[:, 1]) <= 99999) & \
                   (np.abs(points[:, 2]) <= 99999)
    points = points[outlier_mask]
    variance = variance[outlier_mask]
    intensity = intensity[outlier_mask]
    offset_data = offset_data[outlier_mask]
    
    print(f"After removing outliers (>99999): {len(points)} points remain")
    
    # Print final statistics
    print(f"\nFinal point cloud bounds:")
    print(f"X: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
    print(f"Y: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
    print(f"Z: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
    # print(f"\nIntensity statistics:")
    # print(f"Min: {intensity.min()}, Max: {intensity.max()}")
    # print(f"Mean: {intensity.mean():.2f}, Std: {intensity.std():.2f}")
    # print(f"Data type: {intensity.dtype}")
    # print(f"Sample values: {intensity[:10]}")
    # 5. Convert intensity to grayscale colors (0-255 range)
    if intensity.max() > 0:
        colors = (intensity.astype(np.float32) / intensity.max() * 255.0).astype(np.uint8)
    else:
        colors = intensity.astype(np.uint8)
    # Stack to create Nx3 array for RGB (all channels same for grayscale)
    colors = np.stack([colors, colors, colors], axis=1)
    
    return points, colors, intensity, variance, offset_data

def intensity_to_colormap(intensity_norm):
    """
    Convert normalized intensity to RGB colors using a simple colormap.
    
    Args:
        intensity_norm: Normalized intensity values (0-1)
        
    Returns:
        colors: Nx3 array of RGB values (0-1 range)
    """
    # Simple blue -> cyan -> green -> yellow -> red colormap
    colors = np.zeros((len(intensity_norm), 3))
    
    for i, val in enumerate(intensity_norm):
        if val < 0.25:
            # Blue to cyan
            t = val / 0.25
            colors[i] = [0, t, 1]
        elif val < 0.5:
            # Cyan to green
            t = (val - 0.25) / 0.25
            colors[i] = [0, 1, 1 - t]
        elif val < 0.75:
            # Green to yellow
            t = (val - 0.5) / 0.25
            colors[i] = [t, 1, 0]
        else:
            # Yellow to red
            t = (val - 0.75) / 0.25
            colors[i] = [1, 1 - t, 0]
    
    return colors


# # Example usage
# if __name__ == "__main__":
#     # Load the PCD file
#     points, colors, intensity, variance, offset = load_binary_pcd('input.pcd')
    
#     if points is not None:
#         print(f"\nLoaded {len(points)} points")
#         print(f"Point range: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
#               f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
#               f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
#         print(f"Intensity range: [{intensity.min()}, {intensity.max()}]")
#         print(f"Variance range: [{variance.min():.2f}, {variance.max():.2f}]")
#         print(f"Offset range: [{offset.min()}, {offset.max()}]")