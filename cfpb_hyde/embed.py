"""nomic prefixes (DECISIONS N3) and HyDE Eq. 3 averaging."""
import numpy as np

from cfpb_hyde.config import load

_E = load().embedding


def query_text(q: str) -> str:
    return _E.query_prefix + q


def doc_text(d: str) -> str:
    return _E.document_prefix + d


def hyde_vector(q_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Eq. 3 (Gao et al. 2023): v = (f(q) + sum_i f(d_i)) / (N + 1). No extra normalization (DECISIONS H4)."""
    q_vec = np.asarray(q_vec, dtype=np.float32).reshape(1, -1)
    doc_vecs = np.asarray(doc_vecs, dtype=np.float32).reshape(-1, q_vec.shape[1])
    return np.concatenate([q_vec, doc_vecs]).mean(axis=0, keepdims=True)
