# -*- coding: utf-8 -*-
# @Time    : 2018/4/25 20:28
# @Author  : Adesun
# @Site    : https://github.com/Adesun
# @File    : log_parser.py

import argparse
import logging
import os
import platform
import re
import sys
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def get_file_name_and_ext(filename):
    (file_path, temp_filename) = os.path.split(filename)
    (file_name, file_ext) = os.path.splitext(temp_filename)
    return file_name, file_ext


def show_message(message, stop=False):
    print(message)
    if stop:
        sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="training log parser by DeepKeeper ")
    parser.add_argument('--source-dir', dest='source_dir', type=str, default='./',
                        help='the log source directory')
    parser.add_argument('--save-dir', dest='save_dir', type=str, default='./',
                        help='the directory to be saved')
    parser.add_argument('--csv-file', dest='csv_file', type=str, default="",
                        help='training log file')
    parser.add_argument('--log-file', dest='log_file', type=str, default="",
                        help='training log file')
    parser.add_argument('--show', dest='show_plot', type=bool, default=False,
                        help='whether to show')
    return parser.parse_args()


def log_parser(args):
    if not args.log_file:
        show_message('log file must be specified.', True)

    log_path = os.path.join(args.source_dir, args.log_file)
    if not os.path.exists(log_path):
        show_message('log file does not exist.', True)

    file_name, _ = get_file_name_and_ext(log_path)
    log_content = open(log_path).read()

    iterations = []
    losses = []
    fig, ax = plt.subplots()

    major_locator = MultipleLocator(1)
    minor_locator = MultipleLocator(0.5)

    ax.yaxis.set_major_locator(major_locator)
    ax.yaxis.set_minor_locator(minor_locator)

    ax.yaxis.grid(True, which='minor')

    pattern = re.compile(r"([\d]*): .*?, (.*?) avg,")
    matches = pattern.findall(log_content)
    counter = 0
    log_count = len(matches)

    if args.csv_file != '':
        csv_path = os.path.join(args.save_dir, args.csv_file)
        out_file = open(csv_path, 'w')
    else:
        csv_path = os.path.join(args.save_dir, file_name + '.csv')
        out_file = open(csv_path, 'w')

    for match in matches:
        counter += 1
        if log_count > 200:
            if counter % 200 == 0:
                print('parsing {}/{}'.format(counter, log_count))
        else:
            print('parsing {}/{}'.format(counter, log_count))
        iteration, loss = match
        iterations.append(int(iteration))
        losses.append(float(loss))
        out_file.write(iteration + ',' + loss + '\n')

    ax.plot(iterations, losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid()

    save_path = os.path.join(args.save_dir, file_name + '.png')
    plt.savefig(save_path, dpi=300)
    if args.show_plot:
        plt.show()
    else:
        plt.switch_backend('agg')


if __name__ == "__main__":
    args = parse_args()
    log_parser(args)
