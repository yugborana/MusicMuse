import os
import librosa
import soundfile as sf
import numpy as np

input_dir = "data/processed/fma_small" 
output_dir = "data/preprocessed/" 
os.makedirs(output_dir, exist_ok=True)

TARGET_SAMPLE_RATE = 8000
TARGET_DURATION = 5

def preprocess_audio(wav_file):
    print(f"Loading audio: {wav_file}")
    try:
        audio, sr = librosa.load(wav_file, sr=TARGET_SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"Error loading file {wav_file}: {e}")
        return

    audio = audio / np.max(np.abs(audio))

    target_length = TARGET_SAMPLE_RATE * TARGET_DURATION
    if len(audio) > target_length:
        audio = audio[:target_length]  
    else:
        audio = np.pad(audio, (0, target_length - len(audio)))  

    output_file = os.path.join(output_dir, os.path.basename(wav_file))
    try:
        sf.write(output_file, audio, TARGET_SAMPLE_RATE)
        print(f"Processed: {wav_file} -> {output_file}")
    except Exception as e:
        print(f"Error saving file {output_file}: {e}")


def preprocess_all_wav(input_dir, output_dir):
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        
    
        if os.path.isdir(folder_path) and folder.isdigit():
            print(f"Processing folder: {folder}")
            
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    wav_file = os.path.join(folder_path, file)
                    preprocess_audio(wav_file) 

preprocess_all_wav(input_dir, output_dir)
