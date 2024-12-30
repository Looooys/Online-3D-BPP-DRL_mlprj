import torch

# Load the existing data
data = torch.load('scaled_task3.pt')

# Add empty sequence at the beginning
empty_sequence = []  # or torch.tensor([]) depending on your data structure
new_data = [empty_sequence] + data

# Save the modified data
torch.save(new_data, 'scaled_task3_added.pt')