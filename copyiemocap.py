import glob
import os
import shutil
from tqdm import tqdm
from utils import *
def main():
    new_path = config['IEMOCAP']['PATH_TO_AUDIO']
    iemocap_path = config['IEMOCAP']['PATH_TO_DATASET'] 
    for i in tqdm(range(1,6), desc="Processing Sessions"):
        parent_folders = glob.glob(os.path.join(iemocap_path,f"Session{i}/sentences/wav/*"))
        for parent in tqdm(parent_folders, desc=f"Processing folders in Session{i}", leave=False):
            files_list = glob.glob(os.path.join(parent,"*"))
            for file in tqdm(files_list, desc=f"Copying files from {os.path.basename(parent)}", leave=False):
                # Get the filename from the path
                filename = os.path.basename(file)
                # Create the destination path
                dest_path = os.path.join(new_path, filename)
                # Copy the file
                shutil.copy2(file, dest_path)

if __name__ == '__main__':
    main()
