# Downloads necessary python scripts that are not included in repo.
# Author: Wyatt Phillips

import os
from os import path
import urllib.request
import argparse

# Load Configuration
import configparser

config = configparser.ConfigParser()
config.read("settings.ini")
glob = config["DEFAULT"]

def main():
    parser = argparse.ArgumentParser(
        description='Downloads necessary python scripts that are not included in repo.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('url', help='Source URL pointing to file')
    parser.add_argument('--out', default='', help='Target filename to save to')
    args, other_args = parser.parse_known_args()

    url = args.url
    filename = args.url.rsplit('/', 1)[-1] if args.out == '' else args.out

    if not path.exists(filename):
        print(filename + " not found, downloading...")
        try:
            urllib.request.urlretrieve(url, filename)
            print("Downloaded " + filename + " from " + url)
        except Exception as e:
            print(e)
    else:
        print(filename + " exists.")

if __name__ == "__main__":
    main()