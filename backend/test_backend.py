import unittest
import torch
from services.bert_service import bert_service

class TestBertContextService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bert_service.load()

    def test_context_analysis(self):
        sentence_a = "The animal didn't cross the road because it was tired."
        sentence_b = "The animal didn't cross the road because it was raining."
        target_word = "it"

        res = bert_service.analyze_context(
            sentence_a=sentence_a,
            sentence_b=sentence_b,
            target_word=target_word,
            layer=0,
            head=0,
        )

        # Check top-level keys
        self.assertIn("sentence_a", res)
        self.assertIn("sentence_b", res)
        self.assertIn("cosine_similarity", res)
        self.assertIn("static_cosine_similarity", res)
        self.assertEqual(res["target_word"], "it")
        self.assertEqual(res["layer"], 0)
        self.assertEqual(res["head"], 0)
        self.assertEqual(res["hidden_dim"], 768)

        # Check sentence A data
        sa = res["sentence_a"]
        self.assertIn("it", sa["tokens"])
        self.assertEqual(sa["target_token"], "it")
        self.assertEqual(len(sa["hidden_state_10"]), 10)
        self.assertEqual(len(sa["hidden_state_full"]), 768)
        self.assertEqual(len(sa["attention_weights"]), len(sa["tokens"]))
        self.assertEqual(len(sa["attention_matrix"]), len(sa["tokens"]))
        self.assertEqual(len(sa["attention_matrix"][0]), len(sa["tokens"]))
        self.assertAlmostEqual(sa["attention_sum"], 1.0, places=3)
        self.assertFalse(sa["is_uniform"])
        self.assertGreater(sa["attention_std"], 0.0)

        # Check sentence B data
        sb = res["sentence_b"]
        self.assertIn("it", sb["tokens"])
        self.assertEqual(sb["target_token"], "it")
        self.assertEqual(len(sb["hidden_state_10"]), 10)
        self.assertEqual(len(sb["hidden_state_full"]), 768)
        self.assertEqual(len(sb["attention_weights"]), len(sb["tokens"]))
        self.assertAlmostEqual(sb["attention_sum"], 1.0, places=3)
        self.assertFalse(sb["is_uniform"])
        self.assertGreater(sb["attention_std"], 0.0)

        # Check similarity values
        self.assertEqual(res["static_cosine_similarity"], 1.0)
        self.assertLess(res["cosine_similarity"], 0.99)
        self.assertGreater(res["cosine_similarity"], 0.3)

        print("\n[OK] TestBertContextService passed all validations successfully!")
        print(f"Contextual Cosine Similarity: {res['cosine_similarity']}")
        print(f"Static Cosine Similarity: {res['static_cosine_similarity']}")
        print(f"Sentence A tokens ({len(sa['tokens'])}): {sa['tokens']}")
        print(f"Sentence B tokens ({len(sb['tokens'])}): {sb['tokens']}")
        print(f"Sentence A 'it' attention weights: {sa['attention_weights']}")
        print(f"Sentence B 'it' attention weights: {sb['attention_weights']}")

if __name__ == "__main__":
    unittest.main()
