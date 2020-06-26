# Script to store experiment files from runs directory into Google Storage with a given bucket.

import time
import os
from threading import Thread
import sys
import argparse

import torch

def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--path', default='/home/jupyter/digitsum_image/runs', type=str, help='path')
    parser.add_argument('--name', default='digitsum_image_test', type=str, help='name')
    #parser.add_argument('--cur_epoch', default=1, type=int, help='cur epoch')
    #parser.add_argument('--nb_epochs', default=100, type=int, help='nb epochs')
    parser.add_argument('--bucket', default='gs://deepsets', type=str, help="bucket")
    parser.add_argument('--backup-lapse', default=10000, type=int, help='backup lapse')

    args = parser.parse_args()

    return args

def download():
    bashcommand = "sudo zip -r {0}.zip {0}".format(name)
    os.system(bashcommand)
    bashcommand = "gsutil cp {0}.zip {1}".format(name, bucket)
    os.system(bashcommand)

class backup(Thread):

    def run(self):
        while True:
            bashcommand = "sudo zip -r {0}.zip {0}".format(name)
            os.system(bashcommand)
            bashcommand = "gsutil cp {0}.zip {1}".format(name, bucket)
            os.system(bashcommand)
            time.sleep(backup_lapse)

def main():
    global args
    if len(sys.argv) > 1:
        args = parse_args()
        print('----- Backup parameters -----')
        for k, v in args.__dict__.items():
            print(k, ':', v)
    else:
        print('Please provide some parameters for the current experiment. Check-out args.py for more info!')
        sys.exit()

    global path, name, bucket, backup_lapse

    path, name, bucket, backup_lapse = args.path, args.name, args.bucket, args.backup_lapse

    os.chdir(path)
    backup_thread = backup()
    backup_thread.start()

if __name__ == '__main__':
    main()