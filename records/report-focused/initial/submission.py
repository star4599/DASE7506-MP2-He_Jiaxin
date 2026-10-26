"""CPU-only ridge scorer; inference uses only supplied inputs and checkpoint."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from common import PROTOCOL, parse_answer


def candidate_info(bank):
    """Return parsed-answer validity and vote fractions, without references."""
    valid = np.zeros((len(bank), 4), dtype=bool)
    votes = np.zeros((len(bank), 4), dtype=np.float32)
    distinct = np.zeros(len(bank), dtype=np.int32)
    for row_index, row in enumerate(bank):
        answers = [parse_answer(c['text']) for c in row['candidates']]
        counts = Counter(a for a in answers if a is not None)
        distinct[row_index] = len(counts)
        for i, answer in enumerate(answers):
            if answer is not None:
                valid[row_index, i] = True
                votes[row_index, i] = counts[answer] / 4.0
    return valid, votes, distinct


def choose(scores, valid):
    """Exclude unparseable candidates; ties use the earliest candidate."""
    return np.where(valid, scores, -np.inf).argmax(axis=1).tolist()


def embedding_scores(embeddings, mean, scale, weights, bias):
    features = (np.asarray(embeddings, dtype=np.float32) - mean) / scale
    return features @ weights + bias


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Supply --checkpoint pointing to a trained bundle directory.')
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    if config.get('protocol') != PROTOCOL or config.get('format_version') != 1:
        raise ValueError('Unsupported model bundle or protocol.')
    if not bank:
        return []
    with np.load(checkpoint / 'model.npz', allow_pickle=False) as model:
        scores = embedding_scores(
            arrays['embeddings'], model['mean'], model['scale'],
            model['weights'], model['bias'],
        )
    valid, votes, _ = candidate_info(bank)
    scores += np.float32(config['vote_weight']) * votes
    return choose(scores, valid)
