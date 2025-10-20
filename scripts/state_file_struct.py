import torch

state_file = "experiments/MSA-ESRraGANx4plus_ip14+4kv3.2gvWgp3o/training_states/190000.state"
state = torch.load(state_file, map_location="cpu")

# Print the top-level keys
print("Top-level keys in state file:", state.keys())

# Check if the optimizer key exists and inspect its structure
if "optimizers" in state:
    print("Optimizer state keys:", state["optimizers"].keys())
    print("Optimizer param_groups:", state["optimizers"]["param_groups"])
else:
    print("No 'optimizer' key found in state file.")

if "schedulers" in state:
    print("schedulers state keys:", state["schedulers"].keys())
    print("schedulers param_groups:", state["schedulers"]["param_groups"])
else:
    print("No 'optimizer' key found in state file.")