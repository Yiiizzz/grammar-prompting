# neural_lark/code_retriever.py

import numpy as np
from sentence_transformers import SentenceTransformer

from neural_lark.train_utils import logger


class CodeEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading code embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )  # (n, d)
        return emb


class CodeIndex:
    def __init__(self, embeddings: np.ndarray):
        # embeddings 已经是 L2-normalized
        self.embeddings = embeddings

    def search(self, query_emb: np.ndarray, topk: int):
        scores = self.embeddings @ query_emb  # 余弦
        idx = np.argsort(scores)[::-1][:topk]
        return idx


class SymbolMapper:
    """
    用于把草稿里的幻觉函数名映射回最接近的 DSL 函数名。
    """
    def __init__(self, embedder: CodeEmbedder, known_symbols: list[str]):
        self.embedder = embedder
        self.known_symbols = known_symbols
        # 预先对所有 DSL symbol 做 embedding
        self.symbol_embs = embedder.embed(known_symbols)  # (N, d)

    def map_symbol(self, s: str, threshold: float = 0.8):
        """
        把一个 symbol s 映射到已知 DSL symbol 中最相近的一个，
        如果相似度低于 threshold，则返回 None。
        """
        if not s:
            return None
        q_emb = self.embedder.embed(s)[0]  # (d,)
        sims = self.symbol_embs @ q_emb      # (N,)
        idx = int(np.argmax(sims))
        if sims[idx] >= threshold:
            return self.known_symbols[idx]
        else:
            return None

def refine_symbols_with_mapper(symbols, grammar_index, mapper: SymbolMapper):
    symbol_to_rules = grammar_index["symbol_to_rules"]
    refined = []
    for s in symbols:
        if s in symbol_to_rules:
            refined.append(s)
        elif mapper is not None:
            mapped = mapper.map_symbol(s)
            if mapped is not None:
                refined.append(mapped)
    # 去重
    return list(sorted(set(refined)))


def build_code_index_on_targets(train_examples, embedder: CodeEmbedder):
    targets = [ex.target for ex in train_examples]
    logger.info(f"Building code index on {len(targets)} train targets")
    emb = embedder.embed(targets)  # (n_train, d)，已归一化
    return CodeIndex(emb)
