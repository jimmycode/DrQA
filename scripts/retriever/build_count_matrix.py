#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to build the tf-idf document matrices for retrieval."""

import numpy as np
import scipy.sparse as sp
import argparse
import os
import math
import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter

from drqa import retriever
from drqa import tokenizers

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# ------------------------------------------------------------------------------
# Multiprocessing functions
# ------------------------------------------------------------------------------

DOC2IDX = None
PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Build article --> word count sparse matrix.
# ------------------------------------------------------------------------------


def count(ngram, hash_size, doc_id, dtype=np.uint16):
    """Fetch the text of a document and compute hashed ngrams counts."""
    global DOC2IDX
    # Tokenize
    tokens = tokenize(retriever.utils.normalize(fetch_text(doc_id)))

    # Get ngrams from tokens, with stopword/punctuation filtering.
    ngrams = tokens.ngrams(
        n=ngram, uncased=True, filter_fn=retriever.utils.filter_ngram
    )

    # Hash ngrams and count occurences
    counts = Counter([retriever.utils.hash(gram, hash_size) for gram in ngrams])

    # Return in sparse matrix data format.
    row = np.array(list(counts.keys()), dtype=np.uint32)  # Caveat: max hash size math.pow(2,32)=4.2B
    col = np.array([DOC2IDX[doc_id]] * len(counts), np.uint32)  # Caveat: max number of document math.pow(2,32)=4.2B
    data = np.array(list(counts.values()), dtype=dtype)  # use the specified dtype to save memory
    return row, col, data


def get_count_matrix(args, db, db_opts):
    """Form a sparse word to document count matrix (inverted index).

    M[i, j] = # times word i appears in document j.
    """
    # Map doc_ids to indexes
    global DOC2IDX
    db_class = retriever.get_class(db)
    with db_class(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()
    DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    # Setup worker pool
    tok_class = tokenizers.get_class(args.tokenizer)
    workers = ProcessPool(
        args.num_workers,
        initializer=init,
        initargs=(tok_class, db_class, db_opts)
    )

    # Compute the count matrix in steps (to keep in memory)
    logger.info('Mapping...')
    row, col, data = [], [], []
    step = max(int(len(doc_ids) / 10), 1)
    batches = [doc_ids[i:i + step] for i in range(0, len(doc_ids), step)]
    _count = partial(count, args.ngram, args.hash_size, dtype=args.dtype)
    for i, batch in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for b_row, b_col, b_data in workers.imap_unordered(_count, batch):
            row.append(b_row)
            col.append(b_col)
            data.append(b_data)
    workers.close()
    workers.join()

    row = np.concatenate(row)
    col = np.concatenate(col)
    data = np.concatenate(data)

    logger.info('Creating sparse matrix...')
    count_matrix = sp.csr_matrix(
        (data, (row, col)), shape=(args.hash_size, len(doc_ids)), dtype=args.dtype
    )
    count_matrix.sum_duplicates()
    return count_matrix, (DOC2IDX, doc_ids)


def get_doc_lengths(cnts):
    """Return lengths of docs."""
    lengths = np.array(cnts.sum(0)).squeeze()
    return lengths


def get_doc_freqs(cnts):
    """Return word --> # of docs it appears in."""
    binary = (cnts > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', type=str, default=None,
                        help='Path to sqlite db holding document texts')
    parser.add_argument('out_dir', type=str, default=None,
                        help='Directory for saving output files')
    parser.add_argument('--ngram', type=int, default=1,
                        help='Use up to N-size n-grams (e.g. 2 = unigrams + bigrams)')
    parser.add_argument('--hash-size', type=lambda x: int(math.pow(2, int(x))),
                        default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help="String option specifying tokenizer type to use (e.g. 'corenlp')")
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    parser.add_argument('--dtype', type=lambda x: eval(x), default=np.uint16,
                        choices=[np.uint8, np.uint16, np.uint32, np.uint64],
                        help='dtype of sparse matrix (choose the minimum necessary to save memory)')
    args = parser.parse_args()
    logging.info('Arguments: %s' % str(args))

    logging.info('Counting words...')
    count_matrix, doc_dict = get_count_matrix(
        args, 'sqlite', {'db_path': args.db_path}
    )

    logger.info('Getting word-doc frequencies...')
    doc_freqs = get_doc_freqs(count_matrix)

    logger.info('Getting doc lengths...')
    doc_lens = get_doc_lengths(count_matrix)
    avg_doc_len = np.mean(doc_lens)

    basename = os.path.splitext(os.path.basename(args.db_path))[0]
    basename += ('-count-ngram=%d-hash=%d-tokenizer=%s' %
                 (args.ngram, args.hash_size, args.tokenizer))
    filename = os.path.join(args.out_dir, basename)

    logger.info('Saving to %s.npz' % filename)
    metadata = {
        'doc_freqs': doc_freqs,
        'doc_lens': doc_lens,
        'avg_doc_len': avg_doc_len,
        'tokenizer': args.tokenizer,
        'hash_size': args.hash_size,
        'ngram': args.ngram,
        'doc_dict': doc_dict,
    }
    retriever.utils.save_sparse_csr(filename, count_matrix, metadata)