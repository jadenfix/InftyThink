import unittest

from src.eval.answer_extractor import answers_match, batch_evaluate_answers, normalize_answer


class AnswerEvaluationTests(unittest.TestCase):
    def test_latex_fraction_equivalence(self):
        for answer in (r"\frac{1}{2}", r"\boxed{\frac{1}{2}}",
                       r"$\boxed{\tfrac{1}{2}}$", r"\text{\dfrac{1}{2}}"):
            with self.subTest(answer=answer):
                self.assertTrue(answers_match(answer, "0.5"))
                self.assertFalse(answers_match(answer, "0.6"))
        self.assertTrue(answers_match(r"\boxed{\frac{-3}{4}}", "-0.75"))
        self.assertTrue(answers_match(r"\frac{3}{-4}", "-0.75"))

    def test_invalid_outputs_never_count_as_correct(self):
        for answer in ("", " ", "nan", "inf", "-inf", "1e999", None):
            with self.subTest(answer=answer):
                self.assertFalse(answers_match(answer, answer))
                self.assertFalse(answers_match(answer, "0.5"))

    def test_unbalanced_wrappers_are_not_repaired(self):
        self.assertFalse(answers_match(r"\boxed{\frac{1}{2}", "0.5"))

    def test_existing_comparisons(self):
        for a, b in (("1/2", "0.500"), ("1,000,000", "1000000"),
                     (r"\boxed{42}", "42"), ("x + y", "x + y")):
            self.assertTrue(answers_match(a, b))
        self.assertFalse(answers_match("1/0", "0"))
        self.assertFalse(answers_match("2", "3"))

    def test_batch_length_mismatch_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            batch_evaluate_answers(["1", "2"], ["1"])
        self.assertEqual(batch_evaluate_answers(["1", "2"], ["1", "3"]), [True, False])


if __name__ == "__main__":
    unittest.main()
