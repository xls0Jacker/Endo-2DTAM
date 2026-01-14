import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2

# Set base dir to project root
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from utils.slam_helpers import transformed_params2rendervar, transform_to_frame_eval
from utils.recon_helpers import setup_camera
from diff_surfel_rasterization import GaussianRasterizer as Renderer

class GaussianViewer:
    def __init__(self, params_path, resolution=(600, 600), fov_deg=60.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.width, self.height = resolution
        
        # Load params
        print(f"Loading parameters from {params_path}")
        self.params = self._load_params(params_path)
        
        # Camera State
        self.fov = fov_deg
        self.setup_initial_pose()
        
        # Mouse State
        self.mouse_pressed = False
        self.last_x, self.last_y = 0, 0
        self.rot_sensitivity = 0.005
        self.move_speed = 0.05
        
        # Intrinsics (Approximate)
        fx = 0.5 * self.width / np.tan(0.5 * np.deg2rad(self.fov))
        fy = 0.5 * self.height / np.tan(0.5 * np.deg2rad(self.fov))
        self.intrinsics = np.array([
            [fx, 0, self.width/2],
            [0, fy, self.height/2],
            [0, 0, 1]
        ])

    def _load_params(self, path):
        loaded = dict(np.load(path, allow_pickle=True))
        params = {}
        for k, v in loaded.items():
            if isinstance(v, np.ndarray):
                params[k] = torch.tensor(v).to(self.device).float()
            else:
                params[k] = v
        return params

    def setup_initial_pose(self):
        # Try to use the first frame's pose from optimization trace if available
        if 'cam_trans' in self.params and 'cam_unnorm_rots' in self.params:
            try:
                # cam_trans: (1, 3, N), cam_unnorm_rots: (1, 4, N)
                # First frame pose
                # Note: These parameters are usually optimized during SLAM.
                # Assuming params are from a saved checkpoint, they contain the trajectory.
                
                t = self.params['cam_trans'][0, :, 0].cpu().numpy()
                q = self.params['cam_unnorm_rots'][0, :, 0].cpu().numpy() # (w, x, y, z)
                
                q_torch = torch.tensor(q).unsqueeze(0).to(self.device)
                
                # Use project's function if possible
                from utils.slam_external import build_rotation
                R = build_rotation(F.normalize(q_torch)).squeeze().cpu().numpy()
                
                # Endo-2DTAM usually tracks World-to-Camera?
                # transform_to_frame constructs w2c.
                # rel_w2c[:3, :3] = build_rotation(cam_rot)
                # rel_w2c[:3, 3] = cam_tran
                # So params ARE the W2C components.
                
                self.w2c = np.eye(4)
                self.w2c[:3, :3] = R
                self.w2c[:3, 3] = t
                print("Initialized camera from first frame pose.")
                return
            except Exception as e:
                print(f"Could not init from stored pose: {e}")
        
        # Default pose if failed
        print("Using default initial pose.")
        self.w2c = np.eye(4)
        self.w2c[2, 3] = 2.0 # Move back 2 units

    def get_cam_object(self):
        # We transform points manually to the camera frame using transform_to_frame_eval, 
        # so the renderer should use Identity view matrix to render them "as is" in view space.
        return setup_camera(self.width, self.height, self.intrinsics, np.eye(4))

    def render(self):
        with torch.no_grad():
            cam = self.get_cam_object()
            
            # 1. Transform points to current camera frame
            # We use our custom w2c
            w2c_tensor = torch.tensor(self.w2c).float().to(self.device)
            transformed_pts = transform_to_frame_eval(self.params, rel_w2c=w2c_tensor)
            
            # 2. Prepare render variables
            rendervar = transformed_params2rendervar(self.params, transformed_pts)
            
            # 3. Rasterize
            im, _, _ = Renderer(raster_settings=cam)(**rendervar)
            
            # 4. Post-process
            im = torch.clamp(im, 0, 1)
            im_np = im.permute(1, 2, 0).cpu().numpy() # CHW -> HWC
            
            return (im_np * 255).astype(np.uint8)

    def update_pose(self, forward=0, right=0, up=0):
        # We maintain w2c (World to Camera).
        # To move the camera, it's easier to think in Camera-to-World (c2w), move, then invert.
        c2w = np.linalg.inv(self.w2c)
        
        # Camera Frame (OpenGL style usually, but setup_camera uses what?):
        # setup_camera uses standard projection derived from intrinsics.
        # If we render with Identity W2C, valid points must be where?
        # Typically Points in front of camera have +Z or -Z?
        # setup_camera:
        # opengl_proj has [0, 0, 1, 0] in 3rd row -> preserves Z?
        # AND [2*fx/w, ...]
        # It seems it expects Z to be positive for the projection math: w / (2*fx)
        # But wait, glFrustum usually looks down -Z.
        # Let's assume standard CV convention: Z is forward.
        
        R = c2w[:3, :3]
        
        # Move relative to camera orientation axes
        # R[:, 0] is Right, R[:, 1] is Down, R[:, 2] is Forward
        delta = (R[:, 0] * right) + (R[:, 1] * up) + (R[:, 2] * forward)
        c2w[:3, 3] += delta
        
        self.w2c = np.linalg.inv(c2w)

    def rotate_pose(self, dx, dy):
        # Rotation around local X (pitch) and Global/Local Y (yaw)
        c2w = np.linalg.inv(self.w2c)
        
        # Use project helpers for rotation (they expect radians)
        from utils.recon_helpers import rotation_matrix_x, rotation_matrix_y
        
        # dy -> pitch (around X)
        rx = rotation_matrix_x(dy * self.rot_sensitivity)
        # dx -> yaw (around Y)
        ry = rotation_matrix_y(-dx * self.rot_sensitivity)
        
        # Apply pitch locally, yaw globally or locally? 
        # Locally gives "Airplane" feel, Globally gives "FPS" feel.
        # Let's do local for simplicity of matrix math first
        c2w[:3, :3] = c2w[:3, :3] @ ry @ rx
        
        self.w2c = np.linalg.inv(c2w)

    def run(self):
        cv2.namedWindow("Gaussian Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gaussian Viewer", self.width, self.height)
        cv2.setMouseCallback("Gaussian Viewer", self.mouse_callback)
        
        print("\n=== Controls ===")
        print("W/S: Move Forward/Backward")
        print("A/D: Move Left/Right")
        print("Q/E: Move Up/Down")
        print("Mouse Left Drag: Rotate")
        print("ESC: Exit")
        print("================")
        
        while True:
            # Render
            img = self.render()
            
            # Display (RGB to BGR)
            cv2.imshow("Gaussian Viewer", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27: # ESC
                break
            
            # Movement
            f, r, u = 0, 0, 0
            if key == ord('w'): f = self.move_speed
            if key == ord('s'): f = -self.move_speed
            if key == ord('a'): r = -self.move_speed
            if key == ord('d'): r = self.move_speed
            if key == ord('q'): u = -self.move_speed
            if key == ord('e'): u = self.move_speed
            
            if f != 0 or r != 0 or u != 0:
                self.update_pose(forward=f, right=r, up=u)

        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pressed = True
            self.last_x, self.last_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_pressed = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_pressed:
                dx = x - self.last_x
                dy = y - self.last_y
                self.rotate_pose(dx, dy)
                self.last_x, self.last_y = x, y

def main():
    parser = argparse.ArgumentParser(description="Interactive Gaussian Splatting Viewer")
    parser.add_argument("path", help="Path to params.npz file or experiment directory")
    parser.add_argument("--res", type=int, default=600, help="Resolution (square)")
    parser.add_argument("--fov", type=float, default=60.0, help="Field of view")
    args = parser.parse_args()
    
    path = args.path
    if os.path.isdir(path):
        # Try to find params.npz
        potential = os.path.join(path, "params.npz")
        if os.path.exists(potential):
            path = potential
        else:
            # Check for latest paramsX.npz
            try:
                files = os.listdir(path)
                params_files = [f for f in files if f.startswith("params") and f.endswith(".npz")]
                if params_files:
                    import re
                    def extract_num(s):
                        m = re.search(r'\d+', s)
                        return int(m.group()) if m else 0
                    
                    params_files.sort(key=extract_num, reverse=True)
                    path = os.path.join(path, params_files[0])
                    print(f"Auto-selected latest checkpoint: {path}")
                else:
                    print(f"Error: No params.npz found in {path}")
                    return
            except Exception as e:
                print(f"Error accessing directory: {e}")
                return

    if not os.path.exists(path):
        print(f"Error: File not found {path}")
        return
        
    viewer = GaussianViewer(path, resolution=(args.res, args.res), fov_deg=args.fov)
    viewer.run()

if __name__ == "__main__":
    main()
