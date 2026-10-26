"""Final MP2 candidate selector. Offline CPU inference; NumPy is sufficient.

Official entry: predict(bank, arrays, checkpoint). All learned values are loaded
from the checkpoint; no references, stored predictions or question IDs are read.
"""
import ast
from fractions import Fraction
import json
import math
import operator
from pathlib import Path
import re
import numpy as np
from common import PROTOCOL, parse_answer



def question_number_count(question):
    """估计显式数量个数，合并 twenty-five / nine hundred 这类词组。"""
    words = set('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety half quarter twice double triple dozen'.split())
    tens = set('twenty thirty forty fifty sixty seventy eighty ninety'.split())
    units = set('one two three four five six seven eight nine'.split())
    tokens = re.findall('\\d[\\d,]*(?:\\.\\d+)?(?:/\\d+)?|[a-z]+', question.lower())
    index = count = 0
    while index < len(tokens):
        token = tokens[index]
        if token[0].isdigit():
            try:
                Fraction(token.replace(',', ''))
                count += 1
            except (ValueError, ZeroDivisionError):
                pass
        elif token in words:
            count += 1
            if token in tens and index + 1 < len(tokens) and (tokens[index + 1] in units):
                index += 1
            if index + 1 < len(tokens) and tokens[index + 1] == 'hundred':
                index += 1
        index += 1
    return min(count, 5)


def numeric_value(node):
    if isinstance(node, ast.Expression):
        return numeric_value(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return numeric_value(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
    if isinstance(node, ast.BinOp) and type(node.op) in operations:
        result = operations[type(node.op)](numeric_value(node.left), numeric_value(node.right))
        if not math.isfinite(result) or abs(result) > 1000000000000000.0:
            raise ValueError('Numeric range exceeded')
        return result
    raise ValueError('Not a supported numeric expression')


def evaluate_expression(text):
    tree = ast.parse(text.strip(), mode='eval')
    if sum((1 for _ in ast.walk(tree))) > 80:
        raise ValueError('Expression too long')
    return numeric_value(tree)


def candidate_features(bank):
    """先保留七项摘要，再按开关追加六项等式证据。不读取参考答案。"""
    result = []
    for row in bank:
        answers = [parse_answer(c['text']) for c in row['candidates']]
        number_count = question_number_count(row['question'])
        candidates = []
        for candidate, answer in zip(row['candidates'], answers):
            text = candidate['text']
            value = float(Fraction(answer)) if answer is not None else 0.0
            summary = [number_count, math.log1p(abs(value)), float(value < 0), float(answer is not None and value == round(value)), math.log1p(len(text)), math.log1p(text.count('=')), sum((a == answer for a in answers)) / 4 if answer is not None else 0]
            candidates.append(summary)
        result.append(candidates)
    return np.asarray(result, dtype=np.float32)


def answer_types(bank):
    """四类数值答案；无法解析者仍由最终有效性掩码处理。"""
    result = []
    for row in bank:
        kinds = []
        for candidate in row['candidates']:
            answer = parse_answer(candidate['text'])
            value = Fraction(answer) if answer is not None else Fraction(0)
            kinds.append((0 if value.denominator == 1 else 2) + int(value < 0))
        result.append(kinds)
    return np.asarray(result, dtype=np.int32)


def tree_scores(features, weights, initial_scores=None):
    """NumPy 遍历导出的树；不依赖 XGBoost，不缓存题目预测。"""
    flat = features.reshape(-1, features.shape[-1])
    total = np.zeros(len(flat), dtype=np.float32) if initial_scores is None else np.asarray(initial_scores, dtype=np.float32).reshape(-1).copy()
    for root in weights['tree_roots']:
        nodes = np.full(len(flat), root, dtype=np.int32)
        active = np.ones(len(flat), dtype=bool)
        for _ in range(int(weights['tree_max_depth']) + 1):
            rows = np.flatnonzero(active)
            if not len(rows):
                break
            feature = weights['tree_feature'][nodes[rows]]
            leaf = feature < 0
            total[rows[leaf]] += weights['tree_leaf'][nodes[rows[leaf]]]
            active[rows[leaf]] = False
            branch = rows[~leaf]
            if len(branch):
                f = feature[~leaf]
                values = flat[branch, f]
                current = nodes[branch]
                nodes[branch] = np.where(np.isnan(values), weights['tree_missing'][current], np.where(values < weights['tree_threshold'][current], weights['tree_left'][current], weights['tree_right'][current]))
        if active.any():
            raise ValueError('Tree structure exceeds recorded depth')
    return total.reshape(features.shape[:2])


def numeric_expression(segment):
    """Parse the entire segment. Partial matches and implicit products are unknown."""
    segment = segment.strip().rstrip('\\').strip()
    if not segment or len(segment) > 240 or (not re.fullmatch('[0-9.eE()+*/\\-\\s]+', segment)):
        return None
    try:
        value = evaluate_expression(segment)
        return value if math.isfinite(value) else None
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def normalized_line(line):
    line = re.sub('(?<![0-9.])\\*\\*([^*\\n]+)\\*\\*(?![0-9.])', '\\1', line)
    line = line.replace('−', '-').replace('×', '*').replace('÷', '/')
    line = re.sub('(?<=\\d),(?=\\d{3}(?:\\D|$))', '', line)
    for old, new in (('\\times', '*'), ('\\cdot', '*'), ('\\div', '/'), ('\\left', ''), ('\\right', ''), ('\\,', ' '), ('\\!', ' '), ('\\;', ' '), ('\\quad', ' '), ('\\(', ' '), ('\\)', ' '), ('\\[', ' '), ('\\]', ' ')):
        line = line.replace(old, new)
    for _ in range(6):
        changed = re.sub('\\\\(?:d?frac)\\{([0-9. +*/()\\-]+)\\}\\{([0-9. +*/()\\-]+)\\}', '((\\1)/(\\2))', line)
        if changed == line:
            break
        line = changed
    line = re.sub('\\\\boxed\\{([0-9. +*/()\\-]+)\\}', '(\\1)', line)
    return line.replace('$', '')


def arithmetic_evidence(text, details=False):
    checked = wrong = unknown = 0
    maximum = 0.0
    last = None
    late = False
    flags = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        total = raw.count('=')
        if not total:
            offset += len(raw)
            continue
        if any((token in raw for token in ('%', '€', '£', '¥'))) or raw.lstrip().startswith('- '):
            unknown += total
            offset += len(raw)
            continue
        parts = normalized_line(raw).split('=')
        for left, right in zip(parts, parts[1:]):
            a, b = (numeric_expression(left), numeric_expression(right))
            if a is None or b is None:
                unknown += 1
                continue
            checked += 1
            last = b
            error = abs(a - b) / max(1.0, abs(a), abs(b))
            bad = not math.isclose(a, b, rel_tol=1e-06, abs_tol=0.015)
            maximum = max(maximum, error)
            if bad:
                wrong += 1
                late = late or offset >= 0.6 * len(text)
                flags.append(dict(left=left.strip(), right=right.strip(), left_value=a, right_value=b))
        offset += len(raw)
    parsed = parse_answer(text)
    try:
        from fractions import Fraction
        agrees = last is not None and parsed is not None and math.isclose(last, float(Fraction(parsed)), rel_tol=1e-06, abs_tol=0.015)
    except (ValueError, OverflowError):
        agrees = False
    values = np.asarray([math.log1p(text.count('=')), math.log1p(checked), math.log1p(wrong), wrong / max(checked, 1), math.log1p(unknown), min(maximum, 1.0), late, agrees], np.float32)
    return (values, dict(checked=checked, wrong=wrong, unknown=unknown, flags=flags)) if details else values


def answer_groups(bank):
    ids = np.full((len(bank), 4), -1, np.int32)
    mask = np.zeros((len(bank), 4), bool)
    for i, row in enumerate(bank):
        lookup = {}
        for j, candidate in enumerate(row['candidates']):
            answer = parse_answer(candidate['text'])
            if answer is not None:
                ids[i, j] = lookup.setdefault(answer, len(lookup))
        mask[i, :len(lookup)] = True
    return (ids, mask)


def membership(ids):
    return (ids[:, :, None] == np.arange(4)) & (ids[:, :, None] >= 0)


def pool_scores(scores, ids, mode='mean'):
    belongs = membership(ids)
    if mode == 'max':
        return np.where(belongs, scores[:, :, None], -1000000000.0).max(axis=1).astype(np.float32)
    if mode != 'mean':
        raise ValueError('Unsupported answer-group pooling')
    counts = belongs.sum(axis=1)
    return (np.einsum('ncg,nc->ng', belongs.astype(np.float32), scores) / np.maximum(counts, 1)).astype(np.float32)


def softmax(scores, mask):
    safe = np.where(mask, scores, -1000000000.0)
    values = np.exp(safe - safe.max(axis=1, keepdims=True)) * mask
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-30)


def normalize(features, weights, clip=6.0):
    return np.clip((features - weights['feature_mean']) / weights['feature_scale'], -clip, clip).astype(np.float32)


def head_scores(features, weights, limit):
    scores = features @ weights['linear_weight']
    if 'hidden_weight' in weights:
        hidden = np.maximum(features @ weights['hidden_weight'] + weights['hidden_bias'], 0)
        scores += hidden @ weights['output_weight']
    return (limit * np.tanh(scores / limit)).astype(np.float32)


def select_candidates(probabilities, scores, ids):
    selected_group = probabilities.argmax(axis=1)
    eligible = (ids == selected_group[:, None]) & (ids >= 0)
    return np.where(eligible, scores, -np.inf).argmax(axis=1).tolist()


def standardize(embeddings, mean, scale, clip):
    return np.clip((embeddings - mean) / scale, -clip, clip).astype(np.float32)


def numpy_logits(x, weights):
    return x @ weights['linear_weight'] + weights['linear_bias']


def tree_features(bank, x, weights):
    low = x @ weights['projection']
    return np.concatenate([low, low-low.mean(axis=1, keepdims=True),
                           candidate_features(bank)], axis=-1)


def prepare_inputs(bank, embeddings, base, base_config):
    """The frozen 20-D path: 7 text, 5 score/group and 8 arithmetic features."""
    x = standardize(embeddings, base['mean'], base['scale'], base_config['input_clip'])
    scores = numpy_logits(x, base).astype(np.float32)
    scale = base_config.get('base_margin_scale', 1.0)
    scores = tree_scores(tree_features(bank, x, base), base, scores*scale) / scale
    scores += base_config.get('answer_prior_strength', 0.0) * base['answer_type_log_ratio'][answer_types(bank)]
    scores = scores.astype(np.float32)
    ids, mask = answer_groups(bank)
    valid = ids >= 0
    belongs = membership(ids).astype(np.float32)
    counts = belongs.sum(axis=1)
    grouped = np.einsum('ncg,ncd->ngd', belongs, x) / np.maximum(counts, 1)[:, :, None]
    rows = np.arange(len(bank))[:, None]
    locations = np.maximum(ids, 0)
    same = grouped[rows, locations]
    center = (scores*valid).sum(axis=1, keepdims=True) / np.maximum(valid.sum(1), 1)[:, None]
    summary = np.stack([scores-center,
        pool_scores(scores, ids, 'max')[rows, locations]-center,
        pool_scores(scores, ids)[rows, locations]-center,
        np.log1p(counts[rows, locations]), np.square(x-same).mean(axis=-1)], axis=-1)
    arithmetic = np.asarray([[arithmetic_evidence(c['text']) for c in row['candidates']]
                             for row in bank], dtype=np.float32)
    features = np.concatenate([candidate_features(bank), summary, arithmetic], axis=-1).astype(np.float32)
    features[~valid] = 0
    return features, scores, ids, mask


def decision_probabilities(features, scores, ids, mask, weights, config):
    reference = pool_scores(scores, ids, 'max') / config['reference_temperature']
    baseline = softmax(reference, mask)
    normalized = normalize(features, weights, config['feature_clip'])
    probabilities = np.zeros_like(baseline)
    for member in config['members']:
        prefix = member['prefix']
        head = {key[len(prefix):]:value for key,value in weights.items() if key.startswith(prefix)}
        delta = pool_scores(head_scores(normalized, head, config['delta_limit']), ids)
        probabilities += member['weight'] * softmax(reference+delta, mask)
    alpha = config['blend_weight']
    return ((1-alpha)*baseline + alpha*probabilities).astype(np.float32)


def predict(bank, arrays, checkpoint):
    """Return one Python candidate index (0..3) per supplied bank row."""
    if not bank:
        return []
    checkpoint = Path(checkpoint) if checkpoint is not None else Path(__file__).parent/'runs/model_bundle'
    config = json.loads((checkpoint/'config.json').read_text(encoding='utf-8'))
    if (config.get('protocol') != PROTOCOL or config.get('format') != 'new_disagreement_v1'
            or config.get('input_features') != 'evidence_arithmetic'):
        raise ValueError('Expected the frozen final arithmetic-evidence checkpoint.')
    with np.load(checkpoint/'model.npz', allow_pickle=False) as saved:
        weights = {key:saved[key] for key in saved.files}
    base = {key[5:]:value for key,value in weights.items() if key.startswith('base_')}
    features, scores, ids, mask = prepare_inputs(bank, arrays['embeddings'], base, config['base_config'])
    p = decision_probabilities(features, scores, ids, mask, weights, config)
    return select_candidates(p, scores, ids)

