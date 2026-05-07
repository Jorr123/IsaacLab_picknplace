data_recorder.py = Class for data recording into HDF5 format

openvla_agent.py = Class to instantiate openvla model

run_expert_demo.py = Expert controller script

run_openvla_eval.py = OpenVLA evaluation script

stack_cfg.py = Our custom manager-based environment configuration

===================================================================================================================================================================

Put the File in the IsaacLab directory -> IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/my_stack_clean/...

to run the expert controller: ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/my_stack_clean/run_expert_demo.py

to run integrated OpenVLA agent: ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/my_stack_clean/run_openvla_eval.py

(Need to download "pip install transformers==4.38.2 accelerate bitsandbytes timm==0.9.16 sentencepiece imageio[ffmpeg]" to run the OpenVLA eval) 

===================================================================================================================================================================
some changes to be done:

change the    [ custom_path = "/data/openvla_data/pick_place_checkpoint25k_no_agg/pick_place" ]  inside run_openvla_eval.py into the path of the OpenVLA checkpoint

modify the     [ video_writer = imageio.get_writer("/workspace/isaaclab/logs/pick_place_eval_no_agg.mp4", fps=10) ]  inside run_openvla_eval.py into the path of the desired output directory
