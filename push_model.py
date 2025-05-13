import os
import torch
from huggingface_hub import login, create_repo
from utils import HUGGINGFACE_TOKEN, config
from net import UpstreamFinetune

def main():
    """
    Push a pre-trained model to the Hugging Face Hub
    """
    
    # Log in to Hugging Face Hub
    print("Logging in to Hugging Face Hub...")
    login(token=HUGGINGFACE_TOKEN)
    print("Login successful!")

    model_name = "ethereal-resonance-187"
    model_path = os.path.join(config["PATH_TO_SAVED_MODELS"], model_name)
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Initializing model")

    model = UpstreamFinetune.from_pretrained(model_path,config["PATH_TO_PRETRAINED_MODELS"],device=device)

    print("Model load successful")

    remote_path = os.path.join("othsueh", model_name) 
    
    print(f"Creating repository at {remote_path}...")
    create_repo(remote_path)

    print(f"Pushing model to {remote_path}...")
    model.push_to_hub(remote_path)

    print("Upload process completed successfully!")


if __name__ == "__main__":
    main()
