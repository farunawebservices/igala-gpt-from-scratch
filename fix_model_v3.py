import torch
import json
import sys
sys.path.append('scripts')
from model import IgalaGPT, GPTConfig

print("Loading original checkpoint...")
checkpoint = torch.load('outputs/model_checkpoints/igala_gpt_final.pt', 
                       map_location='cpu',
                       weights_only=False)

print("Extracting components...")
# Save state dict separately
torch.save(checkpoint['model_state_dict'], 
          'outputs/model_checkpoints/model_state.pth')

# Save config as JSON
config_dict = {
    'vocab_size': checkpoint['config'].vocab_size,
    'n_embd': checkpoint['config'].n_embd,
    'n_head': checkpoint['config'].n_head,
    'n_layer': checkpoint['config'].n_layer,
    'block_size': checkpoint['config'].block_size,
    'dropout': checkpoint['config'].dropout
}
with open('outputs/model_checkpoints/config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

# Save training logs as JSON
with open('outputs/model_checkpoints/training_logs.json', 'w') as f:
    json.dump(checkpoint['training_logs'], f, indent=2)

print("✅ Saved:")
print("  - model_state.pth (state dict only)")
print("  - config.json")
print("  - training_logs.json")
