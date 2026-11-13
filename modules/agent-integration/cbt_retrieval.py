"""Pure BM25/rank fusion shared by production readout and hosted evaluation.

The caller supplies its frozen dense ranker. This module loads no model and
makes no network requests. The unconfigured product uses lexical retrieval.
"""
from collections import Counter
import math
import re

VERSION = 'cbt-readout-bm25-rrf-v1'


def terms(text):
    return re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', text.casefold())


def bm25(query, documents):
    tokens = [terms(document) for document in documents]
    query_terms = set(terms(query))
    average = sum(map(len, tokens)) / max(1, len(tokens))
    frequencies = Counter(term for row in tokens for term in set(row))
    result = []
    for index, row in enumerate(tokens):
        counts, score = Counter(row), 0.0
        for term in query_terms:
            frequency = counts[term]
            if frequency:
                inverse_frequency = math.log(1 + (len(tokens)-frequencies[term]+0.5)/(frequencies[term]+0.5))
                score += inverse_frequency * frequency*2.2 / (
                    frequency+1.2*(0.25+0.75*len(row)/max(1,average)))
        result.append((index, score))
    return [index for index, _ in sorted(result, key=lambda result: (-result[1],result[0]))]


def fuse(*rankings):
    scores = Counter()
    for ranking in rankings:
        for rank, index in enumerate(ranking,1):
            scores[index] += 1/(60+rank)
    return sorted(scores, key=lambda index: (-scores[index],index))


def rank_documents(query, documents, dense_ranker=None):
    if not documents:
        return []
    lexical = bm25(query, documents)
    if dense_ranker is None:
        return lexical
    dense = list(dense_ranker.rank(query, documents))
    if (any(type(index) is not int for index in dense) or len(dense) != len(documents)
            or set(dense) != set(range(len(documents)))):
        raise ValueError('dense ranking must be a complete permutation of document indexes')
    return fuse(lexical,dense)
