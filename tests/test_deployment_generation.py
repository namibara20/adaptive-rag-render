import os
import unittest
from unittest.mock import patch

import deployment_generation as generation


def evidence(index, text=None):
    return {
        "title": f"Paper {index}",
        "source": "arXiv",
        "source_type": "arxiv",
        "url": f"https://arxiv.org/abs/2401.{index:05d}",
        "paper_id": f"2401.{index:05d}",
        "published": "2024-01-01",
        "text": text or f"Retrieved evidence passage {index}.",
    }


class DeploymentGenerationTests(unittest.TestCase):
    def test_prompt_is_limited_to_top_evidence_and_preserves_metadata(self):
        messages, citations = generation.build_generation_messages(
            "What does the evidence say?", [evidence(i) for i in range(1, 6)]
        )
        combined_prompt = "\n".join(message["content"] for message in messages)
        self.assertEqual(len(messages), 2)
        self.assertEqual(len(citations), 3)
        self.assertIn("Retrieved evidence passage 1", combined_prompt)
        self.assertIn("Retrieved evidence passage 3", combined_prompt)
        self.assertNotIn("Retrieved evidence passage 4", combined_prompt)
        self.assertEqual(citations[0]["sources"][0]["url"], "https://arxiv.org/abs/2401.00001")
        self.assertEqual(citations[0]["sources"][0]["paper_id"], "2401.00001")
        self.assertIn("say the evidence is insufficient", messages[0]["content"])

    def test_generation_result_uses_local_model_and_only_valid_refs(self):
        with patch.object(generation, "_run_with_timeout", return_value="Evidence says X [1]. Invented source [9]."):
            result = generation.generate_grounded_answer("Question", [evidence(1)])
        self.assertIn("[1]", result["answer"])
        self.assertNotIn("[9]", result["answer"])
        self.assertIn("local via llama.cpp", result["model"])
        self.assertEqual(result["citations"][0]["sources"][0]["doi"], None)

    def test_unavailable_generation_fallback_retains_real_source_links(self):
        missing = FileNotFoundError("model unavailable")
        fallback = generation.generation_unavailable_fallback([evidence(2)], missing)
        self.assertIn("Local generation is unavailable", fallback["answer"])
        self.assertIn("FileNotFoundError", fallback["error"])
        self.assertEqual(fallback["citations"][0]["sources"][0]["url"], "https://arxiv.org/abs/2401.00002")

    def test_missing_explicit_model_path_fails_clearly(self):
        with patch.dict(os.environ, {"RAG_QWEN_GGUF_PATH": "Z:/missing/model.gguf"}):
            with self.assertRaisesRegex(FileNotFoundError, "RAG_QWEN_GGUF_PATH"):
                generation._resolve_model_path()


if __name__ == "__main__":
    unittest.main()
