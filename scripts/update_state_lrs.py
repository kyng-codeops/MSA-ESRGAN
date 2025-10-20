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

    # Get learning rate and scheduler from YAML
    lr = yaml_data["train"]["optim_g"]["lr"]
    scheduler = yaml_data["train"]["scheduler"]
    scheduler_type = scheduler["type"]

    # Compute the new learning rate based on the scheduler
    if scheduler_type == "MultiStepLR":
        milestones = scheduler["milestones"]
        gamma = scheduler["gamma"]
        new_lr = lr
        for milestone in milestones:
            if current_iter >= milestone:
                new_lr *= gamma
    elif scheduler_type == "CosineAnnealingLR":
        T_max = scheduler["T_max"]
        eta_min = scheduler["eta_min"]
        new_lr = eta_min + (lr - eta_min) * (1 + torch.cos(torch.tensor(current_iter / T_max * 3.141592653589793))) / 2
    else:
        raise NotImplementedError(f"Scheduler type {scheduler_type} is not supported.")

    # Update the optimizer's learning rate in the state file
    if "optimizers" in state:
        for optimizer_state in state["optimizers"]:
            if "param_groups" in optimizer_state:
                for param_group in optimizer_state["param_groups"]:
                    param_group["lr"] = new_lr
                print(f"Updated optimizer learning rate to: {new_lr}")
            else:
                print("Warning: 'param_groups' not found in optimizer state.")
    else:
        print("Warning: 'optimizers' key not found in the state file.")

    # Update the scheduler milestones in the state file
    if "schedulers" in state:
        for scheduler_state in state["schedulers"]:
            if "milestones" in scheduler_state:
                # Convert milestones from YAML (list) to Counter
                scheduler_state["milestones"] = Counter(scheduler["milestones"])
                print(f"Updated scheduler milestones to: {scheduler_state['milestones']}")
            else:
                print("Warning: 'milestones' not found in scheduler state.")
    else:
        print("Warning: 'schedulers' key not found in the state file.")

    # Save the updated state file
    torch.save(state, output_file)

def main():
    args = parse_args()

    # Load YAML file
    yaml_data = load_yaml(args.opt)

    # Generate the updated state file name
    updated_state_file = args.state.replace(".state", "_updated.state")

    # Update the state file
    update_state_file(args.state, yaml_data, updated_state_file)

    print(f"Updated state file saved to: {updated_state_file}")

if __name__ == "__main__":
    main()