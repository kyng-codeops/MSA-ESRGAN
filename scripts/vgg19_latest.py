import torch
import torchvision.models as models

# Load the VGG19 model with pre-trained weights
vgg19_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT, progress=True)

# Define the path to save the .pth file
save_path = 'vgg19_imagenet_latest.pth'

# Save the state dictionary to the .pth file
torch.save(vgg19_model.state_dict(), save_path)

print(f"VGG19 weights saved to: {save_path}")