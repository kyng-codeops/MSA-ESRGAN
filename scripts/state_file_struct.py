import torch
"""
Script to inspect the structure of a PyTorch state file,
focusing on optimizer and scheduler components for debugging the update_state_lrs.py script.

"""
state_file = "experiments/MSA-ESRraGANx4plus_ip14+4kv3.2gvWgp3ns1a/training_states/130000.state"
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
        # Training ESR-NET has not GAN so only one scheduler exists
        # print("Scheduler state keys:", state["schedulers"].keys())
        for key in state["schedulers"].keys():
            print(f"\t{key}\n")
        print("Scheduler milestones:", state["schedulers"]["milestones"], "\n\n")
    except AttributeError:
        # Training ESR-GAN element 0 tends to be the generator schedule, and element 1 the discriminator.
        for idx, elem in enumerate(state["schedulers"]):
            print(f"Scheduler [{idx}] keys:")
            for key in elem.keys():
                print(f"\t{key}")
            print("\n")
            if "milestones" in elem:
                print(f"\tScheduler [{idx}]['milestones']:\n\t", elem["milestones"], "\n")
                print("\tbase_lrs: {}\n\t_step_count: {}\n\tlast_epoch: {}\n".format(
                    elem["base_lrs"],
                    elem["_step_count"],
                    elem["last_epoch"]
                ))
                print("")
else:
    print("No 'optimizer' key found in state file.", "\n\n")