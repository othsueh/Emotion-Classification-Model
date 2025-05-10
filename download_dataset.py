"""
Hugging Face Dataset Downloader

This script logs in to the Hugging Face Hub and downloads a specified dataset 
to a designated path on your local machine.
"""

import os
import argparse
from datasets import load_dataset
from huggingface_hub import login, hf_hub_download, snapshot_download
from utils import *


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face Hub")
    parser.add_argument(
        "--dataset", 
        required=True,
        help="Name of the dataset to download (e.g., 'squad', 'glue')"
    )
    parser.add_argument(
        "--config", 
        default=None,
        help="Configuration of the dataset (e.g., 'en', 'squad_v2')"
    )
    parser.add_argument(
        "--output_dir", 
        default="../corpus",
        help="Directory to save the dataset"
    )
    parser.add_argument(
        "--method", 
        choices=["datasets", "snapshot", "file"],
        default="datasets",
        help="Method to download the dataset: 'datasets' API, 'snapshot' for whole repo, or 'file' for specific file"
    )
    parser.add_argument(
        "--filename", 
        default=None,
        help="Specific file to download (only used with --method=file)"
    )
    
    return parser.parse_args()


def download_with_datasets_api(dataset_name, config, output_dir):
    """Download using the datasets library API."""
    print(f"Downloading {dataset_name}{f' ({config})' if config else ''} using datasets API...")
    
    # Prepare arguments for load_dataset
    args = {"path": dataset_name, "cache_dir": output_dir}
    if config:
        args["name"] = config
    
    # Load and cache the dataset
    dataset = load_dataset(**args)
    print(f"Dataset downloaded successfully to {output_dir}")
    return dataset


def download_snapshot(dataset_name, output_dir):
    """Download the entire dataset repository."""
    print(f"Downloading entire {dataset_name} repository...")
    
    snapshot_path = snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        local_dir=os.path.join(output_dir, dataset_name.split("/")[-1])
    )
    
    print(f"Repository downloaded successfully to {snapshot_path}")
    return snapshot_path


def download_specific_file(dataset_name, filename, output_dir):
    """Download a specific file from the dataset repository."""
    if not filename:
        raise ValueError("Filename is required when using the 'file' method")
    
    print(f"Downloading file {filename} from {dataset_name}...")
    
    file_path = hf_hub_download(
        repo_id=dataset_name,
        filename=filename,
        repo_type="dataset",
        cache_dir=output_dir
    )
    
    print(f"File downloaded successfully to {file_path}")
    return file_path


def main():
    """
    Main function to execute the download process.
    python download_dataset.py --dataset "othsueh/MSP-Podcast"
    
    """
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log in to Hugging Face Hub
    print("Logging in to Hugging Face Hub...")
    login(token=HUGGINGFACE_TOKEN)
    print("Login successful!")
    
    # Download the dataset based on the selected method
    if args.method == "datasets":
        result = download_with_datasets_api(args.dataset, args.config, args.output_dir)
    elif args.method == "snapshot":
        result = download_snapshot(args.dataset, args.output_dir)
    elif args.method == "file":
        result = download_specific_file(args.dataset, args.filename, args.output_dir)
    
    print("Download process completed successfully!")


if __name__ == "__main__":
    main()