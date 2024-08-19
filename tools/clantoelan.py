
import argparse
from tqdm import tqdm

from stutter.utils.misc import get_eaf_files

from pympi import Elan



def main(args):

    chat_files = get_eaf_files(args.file_path, ext='.cha')

    for key, value in tqdm(chat_files.items()):
        eaf = Elan.eaf_from_chat(value[0])
        eaf.to_file(args.eaf_save_path+key.replace('.cha','.eaf'))
        

if __name__ == '__main__':

    file_path = 'datasets/fluencybank/chat/interview/'
    eaf_save_path = 'datasets/fluencybank/eaf/interview/'

    parser = argparse.ArgumentParser(description='Convert CLAN files to ELAN files')
    parser.add_argument('--file_path', type=str, default=file_path, help='Path to elan files')
    parser.add_argument('--eaf_save_path', type=str, default=eaf_save_path, help='Path to save the combined elan files')
    args = parser.parse_args()

    main(args)