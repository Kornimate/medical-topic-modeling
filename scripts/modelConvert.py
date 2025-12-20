from transformers import AutoModel
from safetensors.torch import save_file
import torch

model_name = "emilyalsentzer/Bio_ClinicalBERT"   # or your BioBERT model

print("Loading model...")
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

print("Preparing contiguous state_dict...")
state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}

output_file = "./model_safetensors/bioclinicalbert.safetensors"

print("Saving as safetensors...")
save_file(state_dict, output_file)

print(f"Done! Saved → {output_file}")
