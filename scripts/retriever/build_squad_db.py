#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to read in and store documents in a sqlite database."""

import argparse
import sqlite3
import json
import os
import logging
import importlib.util

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
from drqa.retriever import utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------

PREPROCESS_FN = None


def init(filename):
    global PREPROCESS_FN
    if filename:
        PREPROCESS_FN = import_module(filename).preprocess


def import_module(filename):
    """Import a module given a full path to the file."""
    spec = importlib.util.spec_from_file_location('doc_filter', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def get_squad(file_path):
    """Iterate over all the SQuAD paragraphs (context)."""

    with open(file_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

    paragraphs = []
    for doc_json in tqdm(dataset):
        title = utils.normalize(doc_json["title"])

        for idx, paragraph_json in enumerate(doc_json['paragraphs']):
            pid = "%s ### %d" % (title, idx)
            text = paragraph_json["context"]
            paragraphs.append((pid, text))

    return paragraphs


def store_contents(squad_path, save_path, num_workers=None):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        squad_path: path to SQuAD dataset (the original SQuAD json format)
          containing json encoded documents (must have `id` and `text` fields).
        save_path: Path to output sqlite db.
        num_workers: Number of parallel processes to use when reading docs.
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text);")

    # workers = ProcessPool(num_workers, initializer=init, initargs=(preprocess,))
    paragraphs = get_squad(squad_path)

    for pid, text in tqdm(paragraphs):
        c.execute("INSERT INTO documents VALUES (?,?)", (pid, text))

    logger.info('Read %d paragraphs.' % len(paragraphs))
    logger.info('Committing...')
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('squad_path', type=str, help='/path/to/data')
    parser.add_argument('save_path', type=str, help='/path/to/saved/db.db')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    store_contents(args.squad_path, args.save_path, args.num_workers)
