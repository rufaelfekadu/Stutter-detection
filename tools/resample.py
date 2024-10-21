import glob
import librosa
import soundfile as sf
import os

import concurrent.futures

def resample_audio(file_path, target_sr=22050):
    try:
        y, sr = librosa.load(file_path, sr=None)
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Create the resampled directory if it doesn't exist
        resampled_dir = os.path.dirname(file_path.replace('wavs', f'wavs_resampled_{target_sr}'))
        os.makedirs(resampled_dir, exist_ok=True)
        
        # Save the resampled file to the new directory
        output_path = os.path.join(resampled_dir, os.path.basename(file_path))
        sf.write(output_path, y_resampled, target_sr)
        print(f"Resampled {file_path} to {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def resample_directory(directory_path, target_sr=22050, num_threads=4):
    audio_files = glob.glob(os.path.join(directory_path, '*.wav'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(resample_audio, file, target_sr) for file in audio_files]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    directory_path = 'datasets/fluencybank/wavs/interview'
    target_sr = 16000  
    # get num processors to use
    num_threads =  os.cpu_count()
    resample_directory(directory_path, target_sr, num_threads)