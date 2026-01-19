import torch
import os
import sys

# Add scripts folder to path so we can import model
sys.path.append('scripts')
from model import IgalaGPT, GPTConfig

print("Starting conversion...")

checkpoint = torch.load('outputs/model_checkpoints/igala_gpt_final.pt', 
                       map_location='cpu',
                       weights_only=False)
print("✅ Loaded checkpoint")

torch.save(checkpoint, 
          'outputs/model_checkpoints/igala_gpt_final_v2.pt',
          _use_new_zipfile_serialization=False,
          pickle_protocol=4)

print("✅ Saved v2")

# Verify
if os.path.exists('outputs/model_checkpoints/igala_gpt_final_v2.pt'):
    size = os.path.getsize('outputs/model_checkpoints/igala_gpt_final_v2.pt')
    print(f"✅ SUCCESS! Size: {size/1000000:.1f} MB")
