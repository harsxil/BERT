import unittest
import torch
from fastapi.testclient import TestClient
from main import app
from services.bert_service import bert_service


class TestBertContextService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bert_service.load()
        cls.client = TestClient(app)

    def test_1_valid_target_it(self):
        """TEST 1: Valid target 'it' in sentence -> VALID"""
        sentence = "The animal didn't cross the road because it was tired."
        target = "it"
        res = bert_service.process_sentence(sentence, target, sentence_label="Sentence A")
        self.assertTrue(res["valid"])
        self.assertEqual(res["target_token"], "it")
        self.assertIn("attention_weights", res)
        self.assertEqual(len(res["attention_weights"]), len(res["tokens"]))

    def test_2_valid_target_casing_IT(self):
        """TEST 2: Target 'IT' (case-insensitive) -> VALID"""
        sentence = "The animal didn't cross the road because it was tired."
        target = "IT"
        res = bert_service.process_sentence(sentence, target, sentence_label="Sentence A")
        self.assertTrue(res["valid"])
        self.assertEqual(res["target_token"], "it")
        self.assertIn("attention_weights", res)

    def test_3_invalid_target_dog(self):
        """TEST 3: Target 'dog' not in sentence -> TARGET_NOT_FOUND"""
        sentence = "The animal didn't cross the road because it was tired."
        target = "dog"
        res = bert_service.process_sentence(sentence, target, sentence_label="Sentence A")
        self.assertFalse(res["valid"])
        self.assertEqual(res["error"], "TARGET_NOT_FOUND")
        self.assertIn("Target word 'dog' was not found", res["message"])
        self.assertNotIn("target_token_index", res)
        self.assertNotIn("attention_weights", res)

    def test_4_invalid_partial_word_ani(self):
        """TEST 4: Target 'ani' inside 'animal' -> TARGET_NOT_FOUND (no partial word matching)"""
        sentence = "The animal is tired."
        target = "ani"
        res = bert_service.process_sentence(sentence, target, sentence_label="Sentence A")
        self.assertFalse(res["valid"])
        self.assertEqual(res["error"], "TARGET_NOT_FOUND")
        self.assertNotIn("target_token_index", res)
        self.assertNotIn("attention_weights", res)

    def test_5_both_sentences_valid(self):
        """TEST 5: Dual sentence analysis with target 'it' in both -> Both valid"""
        sa = "The animal didn't cross the road because it was tired."
        sb = "The animal didn't cross the road because it was raining."
        target = "it"
        res = bert_service.analyze_context(sa, sb, target, layer=0, head=0)
        self.assertTrue(res["valid"])
        self.assertTrue(res["sentence_a"]["valid"])
        self.assertTrue(res["sentence_b"]["valid"])
        self.assertEqual(res["sentence_a"]["target_token"], "it")
        self.assertEqual(res["sentence_b"]["target_token"], "it")
        self.assertIsNotNone(res["cosine_similarity"])
        self.assertIsNotNone(res["static_cosine_similarity"])

    def test_6_both_sentences_invalid(self):
        """TEST 6: Dual sentence analysis with target 'dog' in neither -> Both invalid"""
        sa = "The animal didn't cross the road because it was tired."
        sb = "The animal didn't cross the road because it was raining."
        target = "dog"
        res = bert_service.analyze_context(sa, sb, target, layer=0, head=0)
        self.assertFalse(res["valid"])
        self.assertEqual(res["error"], "TARGET_NOT_FOUND")
        self.assertFalse(res["sentence_a"]["valid"])
        self.assertEqual(res["sentence_a"]["error"], "TARGET_NOT_FOUND")
        self.assertFalse(res["sentence_b"]["valid"])
        self.assertEqual(res["sentence_b"]["error"], "TARGET_NOT_FOUND")
        self.assertIsNone(res["cosine_similarity"])
        self.assertIsNone(res["static_cosine_similarity"])

    def test_7_target_in_only_one_sentence(self):
        """TEST 7: Target appears in sentence A ('tired') but not sentence B ('raining') -> A valid, B TARGET_NOT_FOUND"""
        sa = "The animal didn't cross the road because it was tired."
        sb = "The animal didn't cross the road because it was raining."
        target = "tired"
        res = bert_service.analyze_context(sa, sb, target, layer=0, head=0)
        self.assertFalse(res["valid"])
        self.assertEqual(res["error"], "TARGET_NOT_FOUND")

        # Sentence A is valid and produces analysis
        self.assertTrue(res["sentence_a"]["valid"])
        self.assertEqual(res["sentence_a"]["target_token"], "tired")
        self.assertIn("attention_weights", res["sentence_a"])

        # Sentence B is invalid and produces TARGET_NOT_FOUND with no fallback token
        self.assertFalse(res["sentence_b"]["valid"])
        self.assertEqual(res["sentence_b"]["error"], "TARGET_NOT_FOUND")
        self.assertNotIn("target_token_index", res["sentence_b"])
        self.assertNotIn("attention_weights", res["sentence_b"])
        self.assertIsNone(res["cosine_similarity"])

    def test_api_endpoint_400_on_invalid_target(self):
        """Verify POST /api/analyze-context returns HTTP 400 with structured TARGET_NOT_FOUND"""
        response = self.client.post(
            "/api/analyze-context",
            json={
                "sentence_a": "The animal didn't cross the road because it was tired.",
                "sentence_b": "The animal didn't cross the road because it was raining.",
                "target_word": "dog",
                "layer": 0,
                "head": 0,
            },
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertEqual(data["detail"]["type"], "TARGET_NOT_FOUND")
        self.assertIn("dog", data["detail"]["message"])


if __name__ == "__main__":
    unittest.main()
