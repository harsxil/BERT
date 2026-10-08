"""
BERT Service — loads bert-base-uncased once and performs dual-sentence
contextual representation and self-attention extraction.
"""

import math
import logging
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertModel

logger = logging.getLogger(__name__)


class BertService:
    """Singleton service managing the BERT model and contextual analysis."""

    def __init__(self):
        self.model: Optional[BertModel] = None
        self.tokenizer: Optional[BertTokenizerFast] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False

    def load(self):
        """Load bert-base-uncased model and tokenizer into memory once."""
        if self._loaded:
            return
        logger.info("Loading bert-base-uncased into memory...")
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained(
            "bert-base-uncased",
            output_attentions=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("BERT model successfully loaded on %s", self.device)

    def analyze_context(
        self,
        sentence_a: str,
        sentence_b: str,
        target_word: str,
        layer: int = 0,
        head: int = 0,
    ) -> Dict[str, Any]:
        """
        Analyze both sentences with BERT:
        1. Extract contextual representations (Layer 12 final hidden state) for target_word.
        2. Compute cosine similarity between the two contextual representations.
        3. Compute static word embedding cosine similarity for baseline comparison.
        4. Extract real attention matrices for the requested layer and head.
        5. Extract the target token's attention distribution over all tokens.
        """
        if not self._loaded:
            raise RuntimeError("BERT model not loaded. Call load() first.")

        layer = max(0, min(11, layer))
        head = max(0, min(11, head))

        # Process Sentence A
        res_a = self._process_sentence(sentence_a, target_word, layer, head)

        # Process Sentence B
        res_b = self._process_sentence(sentence_b, target_word, layer, head)

        # Contextual representation cosine similarity (768-dim final hidden states)
        vec_a = res_a["_hidden_tensor"]
        vec_b = res_b["_hidden_tensor"]
        cos_sim = float(F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item())

        # Static word embedding cosine similarity
        static_a = res_a["_static_tensor"]
        static_b = res_b["_static_tensor"]
        static_sim = float(F.cosine_similarity(static_a.unsqueeze(0), static_b.unsqueeze(0)).item())

        # Clean internal tensors from output
        del res_a["_hidden_tensor"]
        del res_a["_static_tensor"]
        del res_b["_hidden_tensor"]
        del res_b["_static_tensor"]

        return {
            "sentence_a": res_a,
            "sentence_b": res_b,
            "target_word": target_word,
            "cosine_similarity": round(cos_sim, 4),
            "static_cosine_similarity": round(static_sim, 4),
            "layer": layer,
            "head": head,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_layer": 12,
            "hidden_dim": 768,
        }

    def _process_sentence(
        self, sentence: str, target_word: str, layer: int, head: int
    ) -> Dict[str, Any]:
        """Run a single sentence through BERT and extract tokens, hidden states, and attention."""
        encoded = self.tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=False,
            truncation=True,
            max_length=128,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Target token mapping using character offsets and subword matching
        target_indices = self._find_target_indices(target_word, sentence, tokens, offsets)
        target_idx = target_indices[0] if target_indices else 1

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Contextual hidden representation (Layer 12 / final hidden state)
        last_hidden = outputs.last_hidden_state[0] # (seq_len, 768)
        target_vec = last_hidden[target_idx] # (768,)

        # Static word embedding from BERT embedding table
        static_table = self.model.embeddings.word_embeddings
        static_vec = static_table(input_ids)[0, target_idx] # (768,)

        # Attention tensor for requested layer and head
        # outputs.attentions is a tuple of 12 tensors, each (1, 12, seq_len, seq_len)
        attn_matrix = outputs.attentions[layer][0, head].cpu().numpy().tolist() # (seq_len, seq_len)
        target_row = attn_matrix[target_idx] # (seq_len,)

        row_tensor = torch.tensor(target_row)
        row_sum = float(row_tensor.sum().item())
        row_min = float(row_tensor.min().item())
        row_max = float(row_tensor.max().item())
        row_std = float(row_tensor.std().item())
        is_uniform = bool((row_max - row_min) < 1e-6)

        return {
            "sentence": sentence,
            "tokens": tokens,
            "target_token": tokens[target_idx],
            "target_token_index": target_idx,
            "target_token_indices": target_indices,
            "hidden_state_10": [round(float(x), 4) for x in target_vec[:10].tolist()],
            "hidden_state_full": [round(float(x), 4) for x in target_vec.tolist()],
            "hidden_dim": int(target_vec.shape[0]),
            "attention_weights": [round(float(x), 4) for x in target_row],
            "attention_matrix": [[round(float(c), 4) for c in r] for r in attn_matrix],
            "attention_sum": round(row_sum, 4),
            "attention_min": round(row_min, 4),
            "attention_max": round(row_max, 4),
            "attention_std": round(row_std, 4),
            "is_uniform": is_uniform,
            "_hidden_tensor": target_vec,
            "_static_tensor": static_vec,
        }

    def _find_target_indices(
        self,
        word: str,
        sentence: str,
        tokens: List[str],
        offsets: List[List[int]],
    ) -> List[int]:
        """Find the token index/indices that correspond to the target word."""
        clean_target = word.lower().strip(".,!?;:'\"")

        # 1. Exact string match against clean tokens
        matches = [
            i for i, tok in enumerate(tokens)
            if tok.lower().replace("##", "") == clean_target
        ]
        if matches:
            return matches

        # 2. Overlap test using character offset ranges
        start_char = sentence.lower().find(clean_target)
        if start_char != -1:
            end_char = start_char + len(clean_target)
            overlap_indices = [
                i for i, (tok, off) in enumerate(zip(tokens, offsets))
                if tok not in ("[CLS]", "[SEP]") and off[1] > start_char and off[0] < end_char
            ]
            if overlap_indices:
                return overlap_indices

        # 3. Fallback: first non-CLS token
        return [1] if len(tokens) > 1 else [0]


# Module singleton
bert_service = BertService()
