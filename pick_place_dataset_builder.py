import os
import glob
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from typing import Iterator, Tuple, Any

class PickPlace2kNoAggohu(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Isaac Lab Peg Insertion task."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release with 7D proprio and dual camera views.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # OpenVLA uses the Universal Sentence Encoder for language conditioning
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata matching the OpenVLA / OpenX spec."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3), # Matches your peg_cfg2.py camera
                            dtype=np.uint8,
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(224, 224, 3), # Matches your peg_cfg2.py wrist camera
                            dtype=np.uint8,
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,), # Your 7D [x,y,z,r,p,y,g] vector
                            dtype=np.float32,
                            doc='Robot state: [x, y, z, roll, pitch, yaw, gripper_width].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,), # Matches your action tensor in run_expert_demo.py
                        dtype=np.float32,
                        doc='Robot action: [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper].',
                    ),
                    'discount': tfds.features.Scalar(dtype=np.float32, doc='Discount factor.'),
                    'reward': tfds.features.Scalar(dtype=np.float32, doc='1.0 on success/final step.'),
                    'is_first': tfds.features.Scalar(dtype=np.bool_, doc='True on first step.'),
                    'is_last': tfds.features.Scalar(dtype=np.bool_, doc='True on last step.'),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_, doc='True on terminal step.'),
                    'language_instruction': tfds.features.Text(doc='Task instruction.'),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding.',
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(doc='Path to the original HDF5 file.'),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits pointing to your 'dataset_peg_insertion' folder."""
        # Update this path to where your .h5 files are stored
        data_path = '/data/episode_*.h5'
        all_episodes = glob.glob(data_path)
        dataset_subset = all_episodes
        
        # Split: 90% Train, 10% Val
        split_idx = int(len(dataset_subset) * 0.9)
        
        return {
            'train': self._generate_examples(episode_paths=dataset_subset[:split_idx]),
            'val': self._generate_examples(episode_paths=dataset_subset[split_idx:]),
        }

    def _generate_examples(self, episode_paths) -> Iterator[Tuple[str, Any]]:
        """Reads HDF5 files from DataRecorder and yields RLDS steps."""
        for ep_path in episode_paths:
            with h5py.File(ep_path, 'r') as f:
           
                lang_instr = f.attrs["language_instruction"]
                lang_embed = self._embed([lang_instr])[0].numpy()
                
         
                images = f['observations/images'][:]
                wrist_images = f['observations/wrist_images'][:]
                proprio = f['observations/proprio'][:]
                actions = f['actions'][:]
                
                episode = []
                num_steps = len(images)
                
                for i in range(num_steps):
                    episode.append({
                        'observation': {
                            'image': images[i],
                            'wrist_image': wrist_images[i],
                            'state': proprio[i].astype(np.float32),
                        },
                        'action': actions[i].astype(np.float32),
                        'discount': 1.0,
                        'reward': float(i == (num_steps - 1)), 
                        'is_first': i == 0,
                        'is_last': i == (num_steps - 1),
                        'is_terminal': i == (num_steps - 1),
                        'language_instruction': lang_instr,
                        'language_embedding': lang_embed,
                    })

                sample = {
                    'steps': episode,
                    'episode_metadata': {'file_path': ep_path}
                }
                
                yield ep_path, sample
