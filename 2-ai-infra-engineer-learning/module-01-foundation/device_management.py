import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = model.to(device)

input_tensor =  input_tensor.to(device)

with torch.no_grad():
    output = model(input_tensor)

output = output.cpu().numpy()