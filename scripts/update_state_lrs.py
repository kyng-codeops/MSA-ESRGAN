import argparse
import yaml
import torch
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Update a training state file with a new learning rate and schedule.\n"
            "The new changes are written to a new state file marked xxxxxx.updated.state\n"
            "where xxxxxx is the original state file iteration.\n\n"
            "You MUST manually modify the yaml to restart from the updated state file.\n"
            "This includes ENSURING net_g and net_d pretrained networks are correctly set.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-state", type=str, required=True, help="Path to the training state file (.state)"
    )
    parser.add_argument(
        "-opt", type=str, required=True, help="Path to the BasicSR YAML options file"
    )
    return parser.parse_args()

def load_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        return yaml.safe_load(f)

def update_state_file(state_file, yaml_data, output_file):
    """
        Use the yaml file starting lr and schedule to compute any
        updated learning rate and milestones based on the current iteration
        so a restart can honor the desired change in training schedule.
    """

    # Load the state file
    state = torch.load(state_file, map_location="cpu")
    current_iter = state.get("iter", 0)  # Use the correct current iteration count

    # Get generator learning rate and scheduler from YAML
    lr_g = yaml_data["train"]["optim_g"]["lr"]
    lr_d = yaml_data["train"]["optim_d"]["lr"]
    scheduler = yaml_data["train"]["scheduler"]
    scheduler_type = scheduler["type"]

    # Compute the new generator learning rate based on the scheduler
    if scheduler_type == "MultiStepLR":
        milestones = scheduler["milestones"]
        gamma = scheduler["gamma"]
        new_lr_g = lr_g
        for milestone in milestones:
            if current_iter >= milestone:
                new_lr_g *= gamma
    elif scheduler_type == "CosineAnnealingLR":
        T_max = scheduler["T_max"]
        eta_min = scheduler["eta_min"]
        new_lr_g = eta_min + (lr_g - eta_min) * (1 + torch.cos(torch.tensor(current_iter / T_max * 3.141592653589793))) / 2
    else:
        raise NotImplementedError(f"Scheduler type {scheduler_type} is not supported.")

    # Compute the new discriminator learning rate based on the scheduler
    if scheduler_type == "MultiStepLR":
        milestones = scheduler["milestones"]
        gamma = scheduler["gamma"]
        new_lr_d = lr_d
        for milestone in milestones:
            if current_iter >= milestone:
                new_lr_d *= gamma
    elif scheduler_type == "CosineAnnealingLR":
        T_max = scheduler["T_max"]
        eta_min = scheduler["eta_min"]
        new_lr_d = eta_min + (lr_d - eta_min) * (1 + torch.cos(torch.tensor(current_iter / T_max * 3.141592653589793))) / 2

    # Update the optimizer's learning rates in the state file
    if "optimizers" in state:
        for idx, optimizer_state in enumerate(state["optimizers"]):
            if "param_groups" in optimizer_state:
                for param_group in optimizer_state["param_groups"]:
                    if idx == 0:
                        # Generator optimizer (index 0)
                        param_group["lr"] = new_lr_g
                        print(f"Updated generator optimizer learning rate to: {new_lr_g}")
                    elif idx == 1:
                        # Discriminator optimizer (index 1)
                        param_group["lr"] = new_lr_d
                        print(f"Updated discriminator optimizer learning rate to: {new_lr_d}")
            else:
                print(f"Warning: 'param_groups' not found in optimizer state [{idx}].")
    else:
        print("Warning: 'optimizers' key not found in the state file.")

    # Update the scheduler milestones in the state file
    if "schedulers" in state:
        for idx, scheduler_state in enumerate(state["schedulers"]):
            if "milestones" in scheduler_state:
                # Convert milestones from YAML (list) to Counter
                scheduler_state["milestones"] = Counter(scheduler["milestones"])
                print(f"Updated scheduler [{idx}] milestones to: {scheduler_state['milestones']}")
            else:
                print(f"Warning: 'milestones' not found in scheduler state [{idx}].")
    else:
        print("Warning: 'schedulers' key not found in the state file.")

    # Save the updated state file
    torch.save(state, output_file)

    # Compute and return the learning rate ratio
    lr_ratio = new_lr_d / new_lr_g if new_lr_g != 0 else 0
    return new_lr_g, new_lr_d, lr_ratio

def main():
    args = parse_args()

    # Load YAML file
    yaml_data = load_yaml(args.opt)

    # Generate the updated state file name
    updated_state_file = args.state.replace(".state", "_updated.state")

    # Update the state file
    new_lr_g, new_lr_d, lr_ratio = update_state_file(args.state, yaml_data, updated_state_file)

    print(f"\nUpdated state file saved to: {updated_state_file}")
    print("\n" + "="*60)
    print("LEARNING RATE SUMMARY")
    print("="*60)
    print(f"Generator (net_g) learning rate:    {new_lr_g:.6e}")
    print(f"Discriminator (net_d) learning rate: {new_lr_d:.6e}")
    print(f"Learning rate ratio (net_d / net_g): {lr_ratio:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()