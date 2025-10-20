import torch
"""
Script to inspect the structure of a PyTorch state file,
focusing on optimizer and scheduler components for debugging the update_state_lrs.py script.

"""
state_file = "experiments/MSA-ESRraGANx4plus_ip14+4kv3.2gvWgp3o/training_states/190000.state"
state = torch.load(state_file, map_location="cpu")

# Print the top-level keys
print("Top-level keys in state file:", state.keys(), "\n\n")

# Check if the optimizer key exists and inspect its structure
if "optimizers" in state:
    try:
        print("Optimizer state keys:", state["optimizers"].keys())
        print("Optimizer param_groups:", state["optimizers"]["param_groups"], "\n\n")
    except AttributeError:
        for elem in state["optimizers"]:
            print("Optimizer element keys:", elem.keys())
            if "param_groups" in elem:
                print("Optimizer param_groups:", elem["param_groups"], "\n\n")
else:
    print("No 'optimizer' key found in state file.", "\n\n")

if "schedulers" in state:
    try:
        print("Scheduler state keys:", state["schedulers"].keys())
        print("Scheduler milestones:", state["schedulers"]["milestones"], "\n\n")
    except AttributeError:
        for elem in state["schedulers"]:
            print("Scheduler element keys:", elem.keys())
            if "milestones" in elem:
                print("Scheduler milestones:", elem["milestones"], "\n\n")
else:
    print("No 'optimizer' key found in state file.", "\n\n")