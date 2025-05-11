import os
import torch
from huggingface_hub import login, create_repo
from utils import HUGGINGFACE_TOKEN, config
from net import UpstreamFinetune, UpstreamFinetuneConfig
from safetensors import safe_open
from safetensors.torch import load_file

def main():
    """
    Push a pre-trained model to the Hugging Face Hub
    """
    
    # Log in to Hugging Face Hub
    print("Logging in to Hugging Face Hub...")
    login(token=HUGGINGFACE_TOKEN)
    print("Login successful!")

    model_name = "clean-jazz-186"
    model_path = os.path.join(config["PATH_TO_SAVED_MODELS"], model_name)
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # First load the config
    print(f"Loading configuration from {model_path}...")
    model_config = UpstreamFinetuneConfig.from_pretrained(model_path)
    
    # Then initialize the model with the config
    print(f"Initializing model with the loaded configuration...")
    model = UpstreamFinetune(
        config=model_config,
        pretrained_path=config["PATH_TO_PRETRAINED_MODELS"]
    )
    
    # Check which format the model is saved in and load accordingly
    model_bin_path = os.path.join(model_path, "pytorch_model.bin")
    model_safetensors_path = os.path.join(model_path, "model.safetensors")
    
    if os.path.exists(model_safetensors_path):
        print(f"Loading model weights from {model_safetensors_path}...")
        state_dict = load_file(model_safetensors_path)
        model.load_state_dict(state_dict)
    elif os.path.exists(model_bin_path):
        print(f"Loading model weights from {model_bin_path}...")
        state_dict = torch.load(model_bin_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"No model weights found at {model_path}. Expected either 'pytorch_model.bin' or 'model.safetensors'")
    
    # Move to device
    print(f"Moving model to device {device}...")
    model = model.to(device)
    print("Model loaded successfully!")

    remote_path = os.path.join("othsueh", model_name) 
    
    print(f"Creating repository at {remote_path}...")
    create_repo(remote_path)

    print(f"Pushing model to {remote_path}...")
    model.push_to_hub(remote_path)

    print("Upload process completed successfully!")


if __name__ == "__main__":
    main()
