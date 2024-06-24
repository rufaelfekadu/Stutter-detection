import subprocess
import os
import json
from tqdm import tqdm
import pandas as pd
import argparse

import re

def get_video_urls(channel_url, ):
    command = f"yt-dlp --flat-playlist -j {channel_url}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error fetching video URLs: {stderr.decode()}")
        return {}

    videos = {}
    for line in stdout.splitlines():
        video_info = json.loads(line)
        # remove everything after the first hyphen
        title = video_info['title'].split('-')[0].replace('Episode', 'ep')
        title = extract_episode_info(title.lower())
        videos[title] = f"https://www.youtube.com/watch?v={video_info['id']}"
  
    return videos


def get_episodes(episods_path):
    # read the csv file
    columns = ['name', 'desc', 'url', 'abbv', 'epid']
    df = pd.read_csv(episods_path, names=columns, header=None)

    # get the strong voices podcast episodes
    df = df[df['abbv']==' StrongVoices']

    # get the episode from the url
    df['episode_name'] = df['url'].apply(lambda x:' '.join(x.split('/')[-1].split('-')[3:]).replace('svpodcast','').replace('episode', 'ep'))
    df['episode_name'] = df['episode_name'].apply(lambda x:extract_episode_info(x.lower()))

    return df['episode_name'].tolist(), df['epid'].tolist()

def download_audio(audio_path, video_url):
    if os.path.exists(f"{audio_path}.mp3"):
        return f"{audio_path}.mp3"
    command = f"yt-dlp -x --audio-format mp3 -o '{audio_path}.%(ext)s' {video_url}"
    process = subprocess.Popen(command, shell=True)
    process.wait()
    return f"{audio_path}.mp3"

def extract_episode_info(text):
    pattern = r'(bonus )?ep ?(\d+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        is_bonus = match.group(1)
        episode = match.group(2)
        return f"{is_bonus.strip() if is_bonus else ''} ep {episode}".strip()
    return None

if __name__ == "__main__":
    # Example usage
    channel_url = "https://www.youtube.com/@strongvoices-overcomingspe152/videos"
    csv_path = "datasets/sep28k/SEP-28k_episodes.csv"

    parser = argparse.ArgumentParser(description='Download audio from youtube')
    parser.add_argument('--channel_url', type=str, default=channel_url,
                        help='URL of the youtube channel')
    parser.add_argument('--audio_path', type=str, default="datasets/sep28k/wavs/StrongVoices/",
                        help='Path to save the audio files')
    parser.add_argument('--episodes', type=str, default=csv_path,
                        help='Path to the csv file containing the episodes')
    
    args = parser.parse_args()

    episode_list, epid_list = get_episodes(args.episodes)

    video_urls = get_video_urls(args.channel_url)
    print(f"Found {len(video_urls)} videos.")

    audio_path = args.audio_path
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)

    for id, episode in zip(epid_list, episode_list):
        
        if not(episode in video_urls.keys()):
            print(f"Skipping {episode}")
            continue

        file_path = os.path.join(audio_path, f"{id}")
        wav_path = f"{file_path}.wav"
        video_url = video_urls[episode]
        file_path = download_audio(file_path, video_url)

        # Convert to 16khz mono wav file
        line = f"ffmpeg -i {file_path} -ac 1 -ar 16000 {wav_path}"
        process = subprocess.Popen([(line)],shell=True)
        process.wait()

        # Remove the original mp3/m4a file
        os.remove(file_path)
