import subprocess
import os
import json
from tqdm import tqdm
import pandas as pd
# Initialize the 404 counter
not_found_count = 0

from fuzzywuzzy import fuzz, process
import re

def extract_numbers(text):
    return re.findall(r'\d+', text)

def custom_scorer(episode, candidate):
    # Standard fuzzy matching score
    standard_score = fuzz.token_sort_ratio(episode, candidate)

    # Extract numbers from both strings
    episode_numbers = extract_numbers(episode)
    candidate_numbers = extract_numbers(candidate)

    # Numerical matching score
    num_score = 0
    if episode_numbers and candidate_numbers:
        common_numbers = set(episode_numbers) & set(candidate_numbers)
        num_score = len(common_numbers) / max(len(episode_numbers), len(candidate_numbers)) * 100

    # Combine scores with more weight on numerical match
    combined_score = 0.7 * standard_score + 0.3 * num_score
    return combined_score

def get_video_urls(channel_url, ):
    command = f"yt-dlp --flat-playlist -j {channel_url}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error fetching video URLs: {stderr.decode()}")
        return []

    videos = {}
    for line in stdout.splitlines():
        video_info = json.loads(line)
        videos[video_info['title']] = f"https://www.youtube.com/watch?v={video_info['id']}"
    return videos

def get_episodes(episods_path):
    # read the csv file
    columns = ['name', 'desc', 'url', 'abbv', 'epid']
    df = pd.read_csv(episods_path, names=columns, header=None)

    # get the strong voices podcast episodes
    df = df[df['abbv']==' StrongVoices']

    # get the episode from the url
    df['episode_name'] = df['url'].apply(lambda x:' '.join(x.split('/')[-1].split('-')[3:]))

    return df['episode_name'].tolist(), df['epid'].tolist()

def find_best_matches(episode_list, scraped_episodes, threshold=80):
    matches = {}
    missed = []
    for episode, epid in zip(*episode_list):
        best_match, score = process.extractOne(episode, scraped_episodes.keys(), scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            matches[epid] = scraped_episodes[best_match]
        else:
            missed.append((episode, epid))
            matches[epid] = None
    return matches, missed


def download_audio(audio_path, video_url):
    global not_found_count
    command = f"yt-dlp -x --audio-format mp3 -o '{audio_path}.%(ext)s' {video_url}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        if b'ERROR: [Errno 404]' in stderr:
            not_found_count += 1
            print(f"404 Not Found for URL: {video_url}")
        else:
            print(f"Error downloading {video_url}: {stderr.decode()}")
        return False
    return True

if __name__ == "__main__":
    # Example usage
    channel_url = "https://www.youtube.com/@strongvoices-overcomingspe152/videos"
    audio_path = "dataset/wavs/StrongVoices/"
    csv_path = "dataset/SEP-28k_episodes.csv"

    episode_list, epid_list = get_episodes(csv_path)

    video_urls = get_video_urls(channel_url)
    print(f"Found {len(video_urls)} videos.")

    #  find the best match for each episode
    best_matchces, missed = find_best_matches((episode_list, epid_list), video_urls)

    print(f"Missed {len(missed)} episodes.")

    if not os.path.exists(audio_path):
        os.makedirs(audio_path)

    for id, video_url in best_matchces.items():
        file_path = os.path.join(audio_path, f"{id}")
        download_audio(file_path, video_url)

    print(f"Total 404 errors: {not_found_count}")
