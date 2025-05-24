import torch
import torchaudio
import pandas as pd
import multiprocessing as mp
import numpy as np
import soundfile as sf
import io
import webdataset as wds
import json
from utils import *
import os

# Global variable
corpus = 'TOTAL'
iemocap_path = config['IEMOCAP']['PATH_TO_AUDIO']
podcast_path = config['MSPPODCAST']['PATH_TO_AUDIO']
# transcript_path = config[corpus]['PATH_TO_TRANSCRIPT']

# Function to process a single shard - moved outside to work with multiprocessing
def process_shard(shard_info):
    shard_idx, start_idx = shard_info
    # Get global variables needed for processing
    global iemocap_path, podcast_path
    
    # Get dataframe from the parent scope
    df = create_webdataset_shards.df
    output_dir = create_webdataset_shards.output_dir
    name = create_webdataset_shards.name
    samples_per_shard = create_webdataset_shards.samples_per_shard
    num_shards = create_webdataset_shards.num_shards
    
    end_idx = min(start_idx + samples_per_shard, len(df))
    shard_df = df.iloc[start_idx:end_idx]
    
    # Format shard number with leading zeros
    shard_name = f"{output_dir}/{name}-samples-{shard_idx:03d}.tar"
    print(f"Creating shard {shard_idx+1}/{num_shards}: {shard_name}")
    
    # Create a shard file
    sink = wds.TarWriter(shard_name)
    
    # Process samples for this shard
    for idx, row in shard_df.iterrows():
        sample_id = row['FileName'].split('.')[0]
        if(row['Corpus'] == 'MSP-IMPROV'):
            continue
        try:
            # Prepare the metadata
            metadata = {
                # 'angry': float(row['angry']),
                # 'sad': float(row['sad']),
                # 'disgust': float(row['disgust']),
                # 'contempt': float(row['contempt']),
                # 'fear': float(row['fear']),
                # 'neutral': float(row['neutral']),
                # 'surprise': float(row['surprise']),
                # 'happy': float(row['happy']),
                'EmoAct': float(row['EmoAct']),
                'EmoVal': float(row['EmoVal']),
                # 'EmoDom': float(row['EmoDom']),
                'dominant_emotion': row['dominant_emotion'],
                # 'SpkrID': row['SpkrID'],
                'gender': row['Gender'],
                'split': row['Split_Set'],
                'corpus': row['Corpus']
            }
            
            # Get audio file path (adjust the pattern based on your file naming)
            if (row['Corpus'] == 'IEMOCAP'):
                audio_file_path = os.path.join(iemocap_path, f"{sample_id}.wav")  # Adjust extension if needed
            elif(row['Corpus'] == 'MSP-PODCAST'):
                audio_file_path = os.path.join(podcast_path, f"{sample_id}.wav")  # Adjust extension if needed


            
            # Get transcription file path
            # trans_path = os.path.join(transcript_path, f"{sample_id}.txt")  # Adjust extension if needed
            
            # if os.path.exists(audio_file_path) and os.path.exists(trans_path):
            if os.path.exists(audio_file_path):
                # Load audio using torchaudio
                waveform, sample_rate = torchaudio.load(audio_file_path)
                
                # Save tensor directly using torch.save
                buffer = io.BytesIO()
                torch.save(waveform, buffer)  # Save just the tensor
                audio_bytes = buffer.getvalue()
                
                # Read the transcription
                # with open(trans_path, 'r', encoding='utf-8') as f:
                #     trans_data = f.read()
                
                # Write to WebDataset
                sample = {
                    "__key__": f"{sample_id}",
                    "audio": audio_bytes,
                    # "text": trans_data,
                    "json": json.dumps(metadata)
                }
                sink.write(sample)
        
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
    sink.close()
    print(f"Completed shard {shard_name} with {len(shard_df)} samples")
    return shard_name

# Update the create_webdataset_shards function to store necessary variables
def create_webdataset_shards(output_dir, name, df, samples_per_shard=4000):
    # Store variables as attributes of the function for access by process_shard
    create_webdataset_shards.df = df
    create_webdataset_shards.output_dir = output_dir
    create_webdataset_shards.name = name
    create_webdataset_shards.samples_per_shard = samples_per_shard
    
    # Calculate number of shards
    num_shards = (len(df) + samples_per_shard - 1) // samples_per_shard
    create_webdataset_shards.num_shards = num_shards
    print(f"Creating {num_shards} shards for {name} dataset")
    
    # Create list of shard information
    shard_infos = [(shard_idx, start_idx) 
                  for shard_idx, start_idx in enumerate(range(0, len(df), samples_per_shard))]
    
    # Process shards in parallel
    with mp.Pool(processes=min(mp.cpu_count(), len(shard_infos))) as pool:
        results = pool.map(process_shard, shard_infos)
    
    print(f"Completed all {len(results)} shards for {name} set")

def main():
    print('='*30 + f"Create webdataset for {corpus}" + '='*30) 

    main_df = pd.read_csv(config[corpus]['PATH_TO_LABEL'])
    output_dir = config[corpus]['PATH_TO_DATASET']
    

    # Test set only
    # split_dir = f"{output_dir}/test"
    # os.makedirs(split_dir, exist_ok=True)
    # create_webdataset_shards(split_dir,"test",main_df,4000)


    # Group by split
    train_df = main_df[main_df['Split_Set'] == 'Train']
    dev_df = main_df[main_df['Split_Set'] == 'Development'] 
    # test_df = main_df[main_df['Split_Set'] == 'Test']

    groups = [
        {
            'name': 'train',
            'df': train_df,
            'samples_per_shard': 4000
        },
        {
            'name': 'validation',
            'df': dev_df,
            'samples_per_shard': 4000
        },
        # {
        #     'name': 'test',
        #     'df': test_df,
        #     'samples_per_shard': 4000
        # },
    ]

    for split in groups:
        split_dir = f"{output_dir}/{split['name']}"
        os.makedirs(split_dir, exist_ok=True)
        create_webdataset_shards(split_dir,split['name'],split['df'],split['samples_per_shard'])


if __name__ == '__main__':
    main()
