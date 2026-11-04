"""
BERT Service — loads bert-base-uncased once and performs dual-sentence
contextual representation and self-attention extraction.
"""

import math
import logging
from typing import Optional, Dict, Any, List, Tuple

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

        # Process Sentence A independently
        res_a = self.process_sentence(sentence_a, target_word, layer, head, sentence_label="Sentence A")

        # Process Sentence B independently
        res_b = self.process_sentence(sentence_b, target_word, layer, head, sentence_label="Sentence B")

        # Validation check across sentences
        if not res_a.get("valid") or not res_b.get("valid"):
            if res_a.get("valid"):
                res_a.pop("_hidden_tensor", None)
                res_a.pop("_static_tensor", None)
            if res_b.get("valid"):
                res_b.pop("_hidden_tensor", None)
                res_b.pop("_static_tensor", None)

            if not res_a.get("valid") and not res_b.get("valid"):
                err_msg = f"Target word '{target_word}' was not found in Sentence A or Sentence B."
            elif not res_a.get("valid"):
                err_msg = f"Target word '{target_word}' was not found in Sentence A."
            else:
                err_msg = f"Target word '{target_word}' was not found in Sentence B."

            return {
                "valid": False,
                "error": "TARGET_NOT_FOUND",
                "message": err_msg,
                "sentence_a": res_a,
                "sentence_b": res_b,
                "target_word": target_word,
                "cosine_similarity": None,
                "static_cosine_similarity": None,
                "layer": layer,
                "head": head,
                "num_layers": 12,
                "num_heads": 12,
                "hidden_layer": 12,
                "hidden_dim": 768,
            }

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
            "valid": True,
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

    def process_sentence(
        self,
        sentence: str,
        target_word: str,
        layer: int = 0,
        head: int = 0,
        sentence_label: str = "sentence",
    ) -> Dict[str, Any]:
        """
        Run a single sentence through BERT and extract tokens, hidden states, and attention.
        If target_word is not found, returns a TARGET_NOT_FOUND error without running target analysis
        or using any fallback token.
        """
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

        # Target token mapping using character offsets and WordPiece boundaries
        valid, target_indices, char_span, msg = self.find_target_indices(
            target_word, sentence, tokens, offsets
        )

        if not valid:
            logger.info("Target: '%s' | Sentence: '%s...' | Target found: False", target_word, sentence[:30])
            return {
                "valid": False,
                "error": "TARGET_NOT_FOUND",
                "message": f"Target word '{target_word}' was not found in {sentence_label}." if sentence_label else f"Target word '{target_word}' was not found in the sentence.",
                "sentence": sentence,
                "tokens": tokens,
                "target_word": target_word,
                "sentence_label": sentence_label,
            }

        target_idx = target_indices[0]
        token_span = (tokens[target_indices[0]], tokens[target_indices[-1]])
        logger.info(
            "Target: '%s' | Sentence: '%s...' | Target found: True | Character span: %s | BERT token span: %s | Token indices: %s",
            target_word,
            sentence[:30],
            char_span,
            token_span,
            target_indices,
        )

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
            "valid": True,
            "sentence": sentence,
            "sentence_label": sentence_label,
            "tokens": tokens,
            "target_token": tokens[target_idx],
            "target_token_index": target_idx,
            "target_token_indices": target_indices,
            "char_span": char_span,
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

    _process_sentence = process_sentence

    def find_target_indices(
        self,
        word: str,
        sentence: str,
        tokens: List[str],
        offsets: List[List[int]],
    ) -> Tuple[bool, List[int], Optional[Tuple[int, int]], str]:
        """
        Identify the BERT token indices corresponding to the target word.
        
        Validates:
        1. Non-empty normalized target word.
        2. Case-insensitive matching consistent with bert-base-uncased.
        3. Word boundary adherence in the sentence (rejects partial matches like 'ani' in 'animal').
        4. Alignment with BERT WordPiece token offsets without partial-token clipping.
        5. Absolute rejection when target is absent — NO fallback index.
        """
        clean_target = word.strip()
        # Strip outer quotation marks if provided (e.g., '"it"' or "'it'")
        if (clean_target.startswith('"') and clean_target.endswith('"')) or \
           (clean_target.startswith("'") and clean_target.endswith("'")):
            clean_target = clean_target[1:-1].strip()

        if not clean_target:
            return False, [], None, "Target word cannot be empty."

        target_lower = clean_target.lower()
        sent_lower = sentence.lower()

        # Find occurrences of target_lower in sent_lower with word boundaries
        matches = []
        start_pos = 0
        while True:
            idx = sent_lower.find(target_lower, start_pos)
            if idx == -1:
                break
            end_idx = idx + len(target_lower)

            # Character boundary check:
            # Preceding char must not be alphanumeric (or match starts at beginning of sentence)
            before_ok = (idx == 0) or not sent_lower[idx - 1].isalnum()
            # Following char must not be alphanumeric (or match ends at end of sentence)
            after_ok = (end_idx == len(sent_lower)) or not sent_lower[end_idx].isalnum()

            if before_ok and after_ok:
                matches.append((idx, end_idx))
            start_pos = idx + 1

        if not matches:
            return False, [], None, f"Target word '{word}' was not found in the sentence."

        # Use first valid occurrence
        target_start, target_end = matches[0]

        # Map to BERT tokens using offsets
        matched_indices = []
        for i, (tok_str, off) in enumerate(zip(tokens, offsets)):
            if tok_str in ("[CLS]", "[SEP]"):
                continue
            if off[0] == off[1] == 0:
                continue
            # Token overlaps with character span
            if off[0] < target_end and off[1] > target_start:
                matched_indices.append(i)

        if not matched_indices:
            return False, [], None, f"Target word '{word}' could not be mapped to BERT tokens."

        # Token boundary integrity check:
        first_off = offsets[matched_indices[0]]
        last_off = offsets[matched_indices[-1]]
        if first_off[0] > target_start or last_off[1] < target_end:
            return False, [], None, f"Target word '{word}' does not align with token boundaries."

        # Ensure target does not cut inside an individual token
        if first_off[0] < target_start or last_off[1] > target_end:
            return False, [], None, f"Target word '{word}' only partially matches a token."

        return True, matched_indices, (target_start, target_end), "OK"


# Module singleton
bert_service = BertService()
