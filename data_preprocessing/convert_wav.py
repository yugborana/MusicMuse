from pydub import AudioSegment
import imageio_ffmpeg
import os

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.converter = ffmpeg_path

input_dir = "data/raw/"
output_dir = "data/processed/"

os.makedirs(output_dir, exist_ok=True)

def convert_mp3_to_wav(mp3_file, output_folder):
    try:
        print(f"Processing: {mp3_file}")

        audio = AudioSegment.from_mp3(mp3_file)

        wav_file = os.path.join(output_folder, os.path.basename(mp3_file).replace(".mp3", ".wav"))

        audio.export(wav_file, format="wav")
        
        print(f"Converted: {mp3_file} -> {wav_file}")

    except Exception as e:
        print(f"Error processing {mp3_file}: {e}")

mp3_files = []
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".mp3"):
            mp3_files.append(os.path.join(root, file))

if not mp3_files:
    print("No MP3 files found in any subdirectories!")

for mp3_file in mp3_files:
    relative_path = os.path.relpath(os.path.dirname(mp3_file), input_dir)
    output_folder = os.path.join(output_dir, relative_path)

    os.makedirs(output_folder, exist_ok=True)

    convert_mp3_to_wav(mp3_file, output_folder)
