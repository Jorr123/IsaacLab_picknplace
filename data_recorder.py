import os
import h5py
import numpy as np
import torch
from datetime import datetime

class DataRecorder:
    def __init__(self, output_dir="data_collection", task_description="insert peg into hole"):
        """
        Initializes the data recorder.
        output_dir: Folder where HDF5 files will be saved.
        task_description: The text instruction for OpenVLA (e.g., "Put the peg in the hole").
        """
        self.output_dir = output_dir
        self.task_desc = task_description
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Buffers to hold data for the current episode
        self.reset_buffers()

    def reset_buffers(self):
        self.images = []
        self.actions = []
        self.proprio = [] 
        self.wrist_images = []

    def add_step(self, obs_dict, action_tensor):
        """
        Records a single timestep.
        obs_dict: The observation dictionary returned by env.step()
        action_tensor: The action tensor sent to env.step()
        """
   
        img_tensor = obs_dict["policy"]["image"][0] # Take env 0
        
        # Move to CPU and Numpy
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.cpu().numpy()
        else:
            img_np = img_tensor

   
        if img_np.dtype == np.float32 or img_np.max() <= 1.5:
            img_np = (img_np * 255).astype(np.uint8)
            
        wrist_image_tensor = obs_dict["policy"]["wrist_image"][0]
        
        if isinstance(wrist_image_tensor, torch.Tensor):
            wrist_img_np = wrist_image_tensor.cpu().numpy()
        else:
            wrist_img_np = wrist_image_tensor


        if wrist_img_np.dtype == np.float32 or wrist_img_np.max() <= 1.5:
            wrist_img_np = (wrist_img_np * 255).astype(np.uint8)
        

        act_np = action_tensor[0].cpu().numpy()
        

        prop_tensor = obs_dict["policy"]["proprio"][0]
        if isinstance(prop_tensor, torch.Tensor):
            prop_np = prop_tensor.cpu().numpy()
        else:
            prop_np = prop_tensor

        self.images.append(img_np)
        self.wrist_images.append(wrist_img_np)
        self.actions.append(act_np)
        self.proprio.append(prop_np)
        # self.proprio.append(prop_np)

    def save_episode(self, ep):
        """
        Saves the currently buffered episode to an HDF5 file.
        """
        if len(self.images) == 0:
            print("[Recorder] Buffer empty, nothing to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = os.path.join(self.output_dir, f"episode_{timestamp}.h5")
        filename = os.path.join(self.output_dir, f"episode_{ep}.h5")

        img_data = np.array(self.images)
        wrist_img_data = np.array(self.wrist_images)
        act_data = np.array(self.actions)
        prop_data = np.array(self.proprio)
        
        with h5py.File(filename, "w") as f:
     
            f.attrs["language_instruction"] = self.task_desc
            
            # Create datasets
            f.create_dataset("observations/images", data=img_data, compression="gzip")
            f.create_dataset("observations/wrist_images", data=wrist_img_data, compression="gzip") # Save here
            f.create_dataset("observations/proprio", data=prop_data) # Add this line
            f.create_dataset("actions", data=act_data)
            


        print(f"[Recorder] Saved episode with {len(self.images)} steps to: {filename}")
        self.reset_buffers()