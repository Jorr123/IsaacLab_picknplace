# run_openvla_eval.py
import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher
import imageio



parser = argparse.ArgumentParser(description="OpenVLA Inference in Isaac Lab")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils
from stack_cfg import StackEnvCfg
from openvla_agent import OpenVLAPolicy 

def main():
   
    print("[INFO] Setting up Isaac Lab Environment...")
    env_cfg = StackEnvCfg()
    env_cfg.scene.num_envs = 1 
    env = ManagerBasedRLEnv(cfg=env_cfg)
    

    custom_path = "/data/openvla_data/pick_place_checkpoint25k_no_agg/pick_place"

   
    agent = OpenVLAPolicy(model_id=custom_path, quantize=True)
    print(f"OpenVLAPolicy is loaded")
   
    obs, _ = env.reset()
    instruction = "pick up the green cube and place it on the red target"

    video_writer = imageio.get_writer("/workspace/isaaclab/logs/pick_place_eval_no_agg.mp4", fps=10)
    
    print(f"[INFO] Starting Inference for task: '{instruction}'")
    curr_ep = 0
    curr_step = 0
    success_count = 0
    failure_count = 0
    MAX_EPISODES = 100
    while simulation_app.is_running():
        
        if curr_ep >= MAX_EPISODES:
            break

        image_tensor = obs["policy"]["image"][0] 
        

        if isinstance(image_tensor, torch.Tensor):
            image_np = image_tensor.cpu().numpy()
        else:
            image_np = image_tensor
            

        if image_np.dtype == np.float32 or image_np.max() <= 1.5:
             image_np = (image_np * 255).astype(np.uint8)


        frame = image_np
        video_writer.append_data(frame)

       
        vla_action = agent.predict_action(image_np, instruction)
        
       
        action_tensor = torch.zeros((env.num_envs, 7), device=env.device)
        
    
        action_tensor[0, :3] = torch.tensor(vla_action[:3], device=env.device)
        
      
        action_tensor[0, 3:6] = torch.tensor(vla_action[3:6], device=env.device)
        
      
        vla_gripper = vla_action[6]
        if vla_gripper < 0.5:
            action_tensor[0, 6] = -1.0 # Close
        else:
            action_tensor[0, 6] = 1.0 # Open
	
        #print(f"action : {action_tensor}")
     
     
        obs, rew, terminated, truncated, info = env.step(action_tensor)

        curr_step += 1
        
        
        if terminated.any() or truncated.any():
            print(f"curr ep: {curr_ep}")
            print(f"curr step: {curr_step}")
            curr_ep += 1

            is_success = terminated[0].item()
            is_timeout = truncated[0].item()

            if is_success:
                success_count += 1
                print(f"[Episode {curr_ep}/{MAX_EPISODES}] RESULT: SUCCESS!")
            elif is_timeout:
                failure_count += 1
                print(f"[Episode {curr_ep}/{MAX_EPISODES}] RESULT: Failure!")


            print("Resetting...")
            #env.reset()

    video_writer.close()
    env.close()

    print("\n=========================================")
    print(f"FINAL EVALUATION RESULTS:")
    print(f"Total Episodes: {curr_ep}")
    print(f"Successes:      {success_count}")
    print(f"Failures:       {failure_count}")
    print(f"Success Rate:   {(success_count/curr_ep)*100:.2f}%")
    print("=========================================\n")


    print("[INFO] Closing Simulation App...")
    simulation_app.close()

if __name__ == "__main__":
    main()
