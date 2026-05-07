# openvla_agent.py
import torch
import numpy as np
import json
import os
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

class OpenVLAPolicy:
    def __init__(self, model_id="openvla/openvla-7b", device="cuda", quantize=True):
        self.device = device
        print(f"[OpenVLA] Loading model: {model_id}...")
        
        # Load Processor
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
    
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            attn_implementation="eager", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True,
        ).to(self.device)
        
  
        print("[OpenVLA] Model loaded successfully.")

        stats_path = os.path.join(model_id, "dataset_statistics.json")


        if os.path.exists(stats_path):
            print(f"[OpenVLA] Loading ALL statistics from: {stats_path}")
            with open(stats_path, 'r') as f:
                full_stats = json.load(f)
      
            dataset_key = None
            for key in full_stats.keys():
                if isinstance(full_stats[key], dict) and "action" in full_stats[key]:
                    dataset_key = key
                    break
            
            if dataset_key is None:
                raise KeyError(f"Could not find valid dataset stats in {stats_path}")

            self.model.config.norm_stats["stats_custom"] = full_stats[dataset_key]
            print(f"[OpenVLA] Successfully injected full stats for '{dataset_key}'")
            
        else:
            raise FileNotFoundError(f"CRITICAL: Could not find {stats_path}. Cannot un-normalize actions!")

    def predict_action(self, image_numpy, instruction: str):

        prompt = f"In: {instruction} Out:"

     
        image_pil = Image.fromarray(image_numpy)
        inputs = self.processor(prompt, image_pil).to(self.device, dtype=torch.bfloat16)

 
        with torch.inference_mode():
            action_tokens = self.model.predict_action(
                **inputs, 
                unnorm_key="stats_custom", 
                do_sample=False  
            )

       
        
        return action_tokens
