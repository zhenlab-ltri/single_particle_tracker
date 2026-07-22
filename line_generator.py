import os
import cv2
import numpy as np
import pandas as pd
import h5py
import re
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from scipy.ndimage import gaussian_filter1d


def extract_boundary_points(binary_mask, boundary_type):
    """
    Extracts the upper boundary of frames of lower mask and the lower boundary of frames of upper mask.
    Args:
    - binary_mask: The particular frame with mask
    - boundary_type: Determines whether to extract the upper edge or the lower edge
    Returns the x and y coordinates of the boundary in a 2 dimensional array (N x 2), the maximum and minimum x value where the mask exists.
    """
    h, w = binary_mask.shape
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    
    has_pixel = np.any(cleaned_mask > 0, axis=0)
    valid_x = np.where(has_pixel)[0]
    
    if len(valid_x) == 0:
        return np.empty((0, 2)), 0, 0

    if boundary_type == 'upper':
        flipped_mask = cleaned_mask[::-1, :]
        first_y_from_bottom = np.argmax(flipped_mask > 0, axis=0)
        valid_y = (h - 1) - first_y_from_bottom[valid_x]
    else:
        valid_y = np.argmax(cleaned_mask > 0, axis=0)[valid_x]
        
    points = np.column_stack((valid_x, valid_y.astype(np.float64)))
    
    sort_idx = np.argsort(points[:, 0])
    points = points[sort_idx]
    
    raw_min_x = points[0, 0]
    raw_max_x = points[-1, 0]
        
    return points, raw_min_x, raw_max_x


import numpy as np

def centerline(upper_pts, u_min_x, u_max_x, lower_pts, l_min_x, l_max_x, num_nodes=2000, trig_harmonics=3):
    """
    Takes a sample of points across the frame and then:
    1) fits a straight baseline from end to end, and 
    2) computes a Fourier series (least squares trigonometric fit) to add any bending 
       using a parameter t to give points as (x(t), y(t)).
    """
    if len(upper_pts) < 15 or len(lower_pts) < 15:
        return None
        
    t_upper = np.linspace(0, 1, len(upper_pts))
    t_lower = np.linspace(0, 1, len(lower_pts))
    t_target = np.linspace(0, 1, num_nodes)
    
    try:
        ux = np.interp(t_target, t_upper, upper_pts[:, 0])
        uy = np.interp(t_target, t_upper, upper_pts[:, 1])
        lx = np.interp(t_target, t_lower, lower_pts[:, 0])
        ly = np.interp(t_target, t_lower, lower_pts[:, 1])
        
        cx = (ux + lx) / 2.0
        cy = (uy + ly) / 2.0
        
        lin_coeff_x = np.polyfit(t_target, cx, deg=1)
        lin_coeff_y = np.polyfit(t_target, cy, deg=1)
        
        base_x = np.polyval(lin_coeff_x, t_target)
        base_y = np.polyval(lin_coeff_y, t_target)
        
        res_x = cx - base_x
        res_y = cy - base_y
        
        A = [np.ones_like(t_target)]
        for h in range(1, trig_harmonics + 1):
            A.append(np.sin(2 * np.pi * h * t_target))
            A.append(np.cos(2 * np.pi * h * t_target))
        A = np.column_stack(A)
        
        coeff_rx, _, _, _ = np.linalg.lstsq(A, res_x, rcond=None)
        coeff_ry, _, _, _ = np.linalg.lstsq(A, res_y, rcond=None)
        
        trig_res_x = A @ coeff_rx
        trig_res_y = A @ coeff_ry
        
        final_x = base_x + trig_res_x
        final_y = base_y + trig_res_y
        
        target_start_x = min(u_min_x, l_min_x)
        target_end_x = max(u_max_x, l_max_x)
        
        vec_prox_x = final_x[0] - final_x[20]
        vec_prox_y = final_y[0] - final_y[20]
        
        extended_proximal = []
        if final_x[0] > target_start_x and vec_prox_x != 0:
            num_ext_nodes = 10
            ext_x = np.linspace(target_start_x, final_x[0], num_ext_nodes, endpoint=False)
            slope_prox = vec_prox_y / vec_prox_x
            ext_y = final_y[0] + (ext_x - final_x[0]) * slope_prox
            extended_proximal = np.column_stack((ext_x, ext_y))
            
        vec_dist_x = final_x[-1] - final_x[-20]
        vec_dist_y = final_y[-1] - final_y[-20]
        
        extended_distal = []
        if final_x[-1] < target_end_x and vec_dist_x != 0:
            num_ext_nodes = 10
            ext_x = np.linspace(final_x[-1], target_end_x, num_ext_nodes + 1)[1:]
            slope_dist = vec_dist_y / vec_dist_x
            ext_y = final_y[-1] + (ext_x - final_x[-1]) * slope_dist
            extended_distal = np.column_stack((ext_x, ext_y))
        
        core_nodes = np.column_stack((final_x, final_y))
        
        blocks = []
        if len(extended_proximal) > 0: 
            blocks.append(extended_proximal)
        blocks.append(core_nodes)
        if len(extended_distal) > 0: 
            blocks.append(extended_distal)
        
        full_skeleton = np.vstack(blocks)
        
        dx = np.diff(full_skeleton[:, 0])
        dy = np.diff(full_skeleton[:, 1])
        step_distances = np.sqrt(dx**2 + dy**2)
        
        cumulative_length = np.zeros(len(full_skeleton))
        cumulative_length[1:] = np.cumsum(step_distances)
        
        total_length = cumulative_length[-1]
        target_spacing = np.linspace(0, total_length, num_nodes)
        
        resampled_x = np.interp(target_spacing, cumulative_length, full_skeleton[:, 0])
        resampled_y = np.interp(target_spacing, cumulative_length, full_skeleton[:, 1])
        
        return np.column_stack((resampled_x, resampled_y))
        
    except Exception as e:
        print(f"Error calculating centerline: {e}")
        return None

def extract_frame_number(filename):
    """
    Extracts frame number from filename.
    Args:
        filename: name of the file containing the masked frame
    Returns:
        The frame number.
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None


def refine_csv(file_path, scale_limit_px=3.0, spatial_sigma=1.2, temporal_sigma=1.5):
    """
    Modifies the coordinates in the csv file as follows:
    1. Uses tanh to compress pixels that are > 3.0px from the major eigenvector  (obtained via SVD) in a single frame.
    2. Filters individual nodes by tracking it across multiple frames to drop major changes caused due to improper masks.
    Args:
        file_path: path to the csv file
        scale_limit_px: threshold pixel value (anything above this from centerline will get damped)
        spatial_sigma: sigma value for gaussian filtering used in step 1
        temporal_sigma: sigma value for gaussian filtering used in step 2
    No return value.
    """
    df = pd.read_csv(file_path)
    refined_rows = []
    
    for frame_id, group in df.groupby('frame_id'):
        sorted_nodes = group.sort_values('node_id').copy()
        pts = sorted_nodes[['x_pixel', 'y_pixel']].to_numpy()
        
        if len(pts) < 5:
            refined_rows.append(sorted_nodes)
            continue
            
        centroid = np.mean(pts, axis=0)
        centered_pts = pts - centroid
        _, _, Vh = np.linalg.svd(centered_pts, full_matrices=False)
        direction_vector = Vh[0]
        direction_vector /= np.linalg.norm(direction_vector)
        
        adjusted_pts = np.zeros_like(pts)
        
        for idx in range(len(pts)):
            pt = pts[idx]
            v = pt - centroid
            
            proj_len = np.dot(v, direction_vector)
            closest_point_on_line = centroid + proj_len * direction_vector
            
            perp_vector = pt - closest_point_on_line
            distance = np.linalg.norm(perp_vector)
            
            if distance > 0.01:
                damped_distance = scale_limit_px * np.tanh(distance / scale_limit_px)
                adjusted_pts[idx] = closest_point_on_line + (perp_vector / distance) * damped_distance
            else:
                adjusted_pts[idx] = pt

        smoothed_x = gaussian_filter1d(adjusted_pts[:, 0], sigma=spatial_sigma, mode='nearest')
        smoothed_y = gaussian_filter1d(adjusted_pts[:, 1], sigma=spatial_sigma, mode='nearest')
        
        sorted_nodes['x_pixel'] = smoothed_x
        sorted_nodes['y_pixel'] = smoothed_y
        refined_rows.append(sorted_nodes)
        
    spatial_df = pd.concat(refined_rows)
    
    temporal_rows = []
    for node_id, group in spatial_df.groupby('node_id'):
        sorted_time = group.sort_values('frame_id').copy()
        
        if len(sorted_time) > 3:
            sorted_time['x_pixel'] = gaussian_filter1d(sorted_time['x_pixel'].to_numpy(), sigma=temporal_sigma, mode='nearest')
            sorted_time['y_pixel'] = gaussian_filter1d(sorted_time['y_pixel'].to_numpy(), sigma=temporal_sigma, mode='nearest')
            
        temporal_rows.append(sorted_time)
        
    final_df = pd.concat(temporal_rows).sort_values(['frame_id', 'node_id'])
    final_df.to_csv(file_path, index=False)
    

def create_csv(folders, boundary, input_path, output_path):
    """
    Writes a csv file that is (N * num_nodes) x 5, each row containing 
    frame number, node id, x coordinate, y coordinate, and the global node spacing.
    Args:
        folders: paths to folders with the upper masks and lower masks
        boundary: ['upper', 'lower'] for computing the specific edge of each mask
        input_path: path to the original h5 video
        output_path: path to the folder to which the csv gets saved
    No return value.
    """
    with h5py.File(input_path, 'r') as h5_in:
        detected_video_key = list(h5_in.keys())[0]
        total_video_frames, img_height, width = h5_in[detected_video_key].shape[:3]

    upper_idx = boundary.index('upper') if 'upper' in boundary else None
    lower_idx = boundary.index('lower') if 'lower' in boundary else None

    upper_files = sorted([f for f in os.listdir(folders[upper_idx]) if f.endswith(('.png', '.jpg'))])
    lower_files = sorted([f for f in os.listdir(folders[lower_idx]) if f.endswith(('.png', '.jpg'))])

    with open(output_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_id', 'node_id', 'x_pixel', 'y_pixel', 'node_spacing'])
    
        for u_file, l_file in zip(upper_files, lower_files):
            frame_idx = extract_frame_number(u_file)
            if frame_idx is None or frame_idx >= total_video_frames:
                continue
                
            img_u = cv2.imread(os.path.join(folders[upper_idx], u_file), cv2.IMREAD_GRAYSCALE)
            img_l = cv2.imread(os.path.join(folders[lower_idx], l_file), cv2.IMREAD_GRAYSCALE)
            
            if img_u is None or img_l is None:
                continue
                
            _, mask_u = cv2.threshold(img_u, 1, 255, cv2.THRESH_BINARY)
            _, mask_l = cv2.threshold(img_l, 1, 255, cv2.THRESH_BINARY)
            
            upper_pts, u_min, u_max = extract_boundary_points(mask_u, 'upper')
            lower_pts, l_min, l_max = extract_boundary_points(mask_l, 'lower')
            
            skeleton_nodes = centerline(upper_pts, u_min, u_max, lower_pts, l_min, l_max, num_nodes=2000, trig_harmonics=3)
            
            if skeleton_nodes is not None:
                dx = np.diff(skeleton_nodes[:, 0])
                dy = np.diff(skeleton_nodes[:, 1])
                step_distances = np.sqrt(dx**2 + dy**2)
                node_spacing = np.mean(step_distances)
                
                for node_idx, (x_val, y_val) in enumerate(skeleton_nodes):
                    safe_x = np.clip(x_val, 0, width - 1)
                    safe_y = np.clip(y_val, 0, img_height - 1)
                    csv_writer.writerow([frame_idx, node_idx, f"{safe_x:.4f}", f"{safe_y:.4f}", f"{node_spacing:.6f}"])


def create_overlay_video(input_source, csv_coordinates_path, output_video_path, fps=30):
    df_skeleton = pd.read_csv(csv_coordinates_path)
    skeleton = {frame_id: group[['x_pixel', 'y_pixel']].to_numpy() for frame_id, group in df_skeleton.groupby('frame_id')}
    
    with h5py.File(input_source, 'r') as h5_in:
        video_key = list(h5_in.keys())[0]
        dataset = h5_in[video_key]
        total_frames, img_height, width = dataset.shape[:3]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, img_height))
        
        valid_frames = sorted(skeleton.keys())
        
        for frame_idx in valid_frames:
            if frame_idx >= total_frames or frame_idx < 0:
                continue
                
            raw_frame = np.array(dataset[frame_idx])
            color_frame = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
            
            nodes = skeleton[frame_idx]
            for i in range(len(nodes) - 1):
                p1 = (int(round(nodes[i][0])), int(round(nodes[i][1])))
                p2 = (int(round(nodes[i+1][0])), int(round(nodes[i+1][1])))
                if (0 <= p1[0] < width and 0 <= p1[1] < img_height and 
                    0 <= p2[0] < width and 0 <= p2[1] < img_height):
                    cv2.line(color_frame, p1, p2, (0, 0, 255), 1)
                    
            video_writer.write(color_frame)
            
        video_writer.release()


class SkeletonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Centerline Generator")
        self.root.geometry("700x250")
        self.upper_folder = tk.StringVar()
        self.lower_folder = tk.StringVar()
        self.hdf5_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.create_widgets()

    def create_widgets(self):
        paths_frame = ttk.LabelFrame(self.root, text="Configurations", padding=10)
        paths_frame.pack(fill="x", padx=15, pady=8)
        
        self.add_path_row(paths_frame, "Upper Mask Folder:", self.upper_folder, is_dir=True)
        self.add_path_row(paths_frame, "Lower Mask Folder:", self.lower_folder, is_dir=True)
        self.add_path_row(paths_frame, "Original Video (HDF5 / .h5):", self.hdf5_path, is_dir=False, file_types=[("HDF5 files", "*.h5 *.hdf5")])
        self.add_path_row(paths_frame, "Output Directory:", self.output_dir, is_dir=True)

        run_frame = ttk.Frame(self.root, padding=5)
        run_frame.pack(fill="x", padx=15, pady=5)
        self.btn_run = ttk.Button(run_frame, text="Generate Centerline", command=self.start_processing_thread)
        self.btn_run.pack(fill="x", ipady=6)

    def add_path_row(self, parent, label_text, var, is_dir, file_types=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label_text, width=28, anchor="w").pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=5)
        btn_text = "Browse Folder" if is_dir else "Browse File"
        ttk.Button(row, text=btn_text, command=lambda: self.browse_path(var, is_dir, file_types)).pack(side="right")

    def browse_path(self, var, is_dir, file_types):
        path = filedialog.askdirectory() if is_dir else filedialog.askopenfilename(filetypes=file_types)
        if path: var.set(path)

    def start_processing_thread(self):
        if not self.upper_folder.get() or not self.lower_folder.get():
            messagebox.showerror("Error", "Both upper and lower background mask folders are required.")
            return
        if not self.hdf5_path.get() or not self.output_dir.get():
            messagebox.showerror("Error", "Verify execution paths.")
            return
        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def run_pipeline(self):
        self.btn_run.configure(state="disabled")
        
        folders = [self.upper_folder.get(), self.lower_folder.get()]
        boundary = ['upper', 'lower']

        output_csv_path = os.path.join(self.output_dir.get(), 'pharynx_axis_coordinates.csv')
        output_video_path = os.path.join(self.output_dir.get(), 'pharynx_skeleton_overlay.mp4')
        
        try:
            create_csv(folders, boundary, self.hdf5_path.get(), output_csv_path)
            refine_csv(output_csv_path, scale_limit_px=3.0, spatial_sigma=1.2, temporal_sigma=1.5)
            create_overlay_video(self.hdf5_path.get(), output_csv_path, output_video_path)
            messagebox.showinfo("Success", "Centerline saved to output directory.")
        except Exception as e:
            messagebox.showerror("Execution Fault", f"Pipeline failed:\n{str(e)}")
        finally:
            self.btn_run.configure(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = SkeletonApp(root)
    root.mainloop()
