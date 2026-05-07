import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Expert Demo for Pick and Place")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch!
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.utils.math as math_utils  # <--- CRITICAL IMPORT FOR QUATERNIONS
from isaaclab.envs import ManagerBasedRLEnv
from data_recorder import DataRecorder


from stack_cfg import StackEnvCfg


def flush_camera_pipeline(env):
    """
    Feeds zero-velocity actions to let the camera rendering buffer 
    catch up to the physics teleports after a reset.
    """
    zero_action = torch.zeros((env.num_envs, 7), device=env.device)
    zero_action[:, 6] = 1.0 
    
    for _ in range(2):
        obs, _, _, _, _ = env.step(zero_action)
        
    return obs

def main():
    print("[INFO] Setting up Expert Demo...")
    env_cfg = StackEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print("[INFO] Resetting environment...")
    obs, _ = env.reset()

    robot = env.scene["robot"]
    cube = env.scene["cube3"]
    target = env.scene["target"]


    recorder = DataRecorder(
        output_dir="/data/openvla_data/dataset_pick_place_3", 
        task_description="pick up the yellow cube and place it on the red target"
    )

    print("[INFO] Starting Simulation Loop...")
    
    step_count = 0
    max_episodes = 3000
    curr_ep = 0
    saved_eps = 2000

    phase = 0
    wait_steps = 0

    while simulation_app.is_running():
        if saved_eps >= max_episodes:
            break
 
        cube_pos = cube.data.root_pos_w[0, :3].squeeze()
        target_pos = target.data.root_pos_w[0, :3].squeeze()
        
        hand_body_idx = robot.find_bodies("panda_hand")[0]
        hand_pos = robot.data.body_pos_w[0, hand_body_idx, :3].squeeze()
        hand_quat = robot.data.body_quat_w[0, hand_body_idx, :].squeeze() 


        action = torch.zeros((env.num_envs, 7), device=env.device)
        
       
        if step_count == 0:
            phase = 0
            wait_steps = 0
            
           
            z_random = (torch.rand(1).item() * 0.01)
            xy_random = (torch.randn(2, device=env.device) * 0.005)


        target_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device)
        quat_err = math_utils.quat_mul(target_quat, math_utils.quat_conjugate(hand_quat))
        rot_vec = math_utils.axis_angle_from_quat(quat_err)
        action[0, 3:6] = rot_vec * 1.5

        dist_to_cube_xy = torch.norm(cube_pos[:2] - hand_pos[:2])
        dist_to_target_xy = torch.norm(target_pos[:2] - hand_pos[:2])


        if phase == 0:
            action[0, 6] = 1.0  
            action[0, 0:2] = (cube_pos[:2] - hand_pos[:2] + xy_random) * 2.0
            
            target_height = cube_pos[2] + 0.175 
            action[0, 2] = (target_height + z_random - hand_pos[2]) *1.1

            if dist_to_cube_xy < 0.015 and abs(target_height - hand_pos[2]) < 0.02:
                phase = 1

        elif phase == 1:
           
            action[0, 6] = 1.0  
            action[0, 0:2] = (cube_pos[:2] - hand_pos[:2]) * 3.0  
            action[0, 2] = -0.05 
            # print(f"hand: {hand_pos[2]}")
            # print(f"cube: {cube_pos[2]}")
           
            diff = abs(hand_pos[2] - cube_pos[2])
            # print(f"diff: {diff}")
            if hand_pos[2] < cube_pos[2] + 0.01 or diff < 0.12:
                phase = 2
                wait_steps = 0

        elif phase == 2:
  
            action[0, 6] = -1.0  
            action[0, 0:3] = 0.0 
            
            wait_steps += 1
            if wait_steps > 10:  
                phase = 3

        elif phase == 3:
          
            action[0, 6] = -1.0  
            action[0, 0:2] = 0.0 
            action[0, 2] = 0.06 
            
            if hand_pos[2] > 0.175: 
                phase = 4

        elif phase == 4:
          
            action[0, 6] = -1.0  
            action[0, 0:2] = (target_pos[:2] - hand_pos[:2]) * 2.0
            action[0, 2] = (0.175 - hand_pos[2]) 
            
            if dist_to_target_xy < 0.015:
                phase = 5

        elif phase == 5:
          
            action[0, 6] = -1.0 
            action[0, 0:2] = (target_pos[:2] - hand_pos[:2]) * 3.0
            action[0, 2] = -0.04 
            diff = abs(hand_pos[2] - target_pos[2])
            if hand_pos[2] < target_pos[2] + 0.035 or diff < 0.14: 
                phase = 6
                wait_steps = 0

        elif phase == 6:
          
            action[0, 6] = 1.0 
            action[0, 0:3] = 0.0 
            
            wait_steps += 1
            if wait_steps > 10:
                phase = 7 

        elif phase == 7:
      
            action[0, 6] = 1.0
            action[0, 2] = 0.05
           
        if phase in [0, 1, 4]:
            noise = torch.randn(3, device=env.device) * 0.001
            action[0, 0:3] += noise
            
        # 5. Record and Step
        recorder.add_step(obs, action)
        obs, rew, terminated, truncated, info = env.step(action)
        
        step_count += 1
        
        if terminated.any() or truncated.any():
            is_success = terminated[0].item()
            if is_success:
                print(f"[INFO] Episode {saved_eps} done. Resetting...")
                #recorder.save_episode(saved_eps)
                saved_eps += 1

            else:
                recorder.reset_buffers()
                print(f"[Episode {curr_ep}] TIMEOUT — discarded.")
            
            obs, _ = env.reset()
            obs = flush_camera_pipeline(env)
            
            
            step_count = 0
            curr_ep += 1

    env.close()
    print(f"\n[INFO] Done. Saved {saved_eps} successful episodes to 'dataset_pick_place/'.")

if __name__ == "__main__":
    main()