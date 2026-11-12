"""BM25 and immutable E5 source rankings, with matched whole-source packing."""
from collections import Counter
import math
import re

DENSE_MODEL = "intfloat/multilingual-e5-small"
DENSE_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


from pathlib import Path
import sys

INTEGRATION = Path(__file__).resolve().parents[3] / "modules/agent-integration"
sys.path.insert(0, str(INTEGRATION))
from cbt_retrieval import terms, bm25, fuse, rank_documents


class DenseRetriever:
    """Load only in hosted Colab. Encoder and returned revision are frozen."""
    def __init__(self):
        from auto_eval_v2.run_colab import require_hosted
        require_hosted()
        from sentence_transformers import SentenceTransformer
        self.model_id, self.revision = DENSE_MODEL, DENSE_REVISION
        self.model = SentenceTransformer(DENSE_MODEL, revision=DENSE_REVISION,
                                         trust_remote_code=False, device="cpu")
        self.cache = {}

    def rank(self, query, documents):
        import numpy as np
        missing = list(dict.fromkeys(d for d in documents if d not in self.cache))
        if missing:
            vectors = self.model.encode(["passage: "+d for d in missing], normalize_embeddings=True,
                                        show_progress_bar=False)
            self.cache.update(zip(missing, vectors))
        q = self.model.encode(["query: "+query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = np.asarray([float(self.cache[d] @ q) for d in documents])
        return sorted(range(len(documents)), key=lambda i: (-scores[i], i))


def rank(query, sources, dense):
    documents = [s["content"] for s in sources]
    return rank_documents(query, documents, dense) if documents else []
