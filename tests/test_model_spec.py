import unittest

from backends import ModelSpec


class ModelSpecTestCase(unittest.TestCase):
    def test_empty_fully_contains_empty_is_true(self):
        a = ModelSpec()
        b = ModelSpec()
        self.assertTrue(b.fully_contains(a))

    def test_b_fully_contains_empty_is_true(self):
        a = ModelSpec()
        b = ModelSpec(model_name="model_b")
        self.assertTrue(b.fully_contains(a))

    def test_b_fully_contains_a_with_different_attr_is_false(self):
        a = ModelSpec(model_name="model_a")
        b = ModelSpec(model_name="model_b")
        self.assertFalse(b.fully_contains(a))

    def test_b_fully_contains_a_with_partially_different_attr_is_false(self):
        a = ModelSpec(model_name="model_a", backend="backend_a")
        b = ModelSpec(model_name="model_a", backend="backend_b")
        self.assertFalse(b.fully_contains(a))

    def test_b_fully_contains_a_with_partially_matching_attr_is_true(self):
        a = ModelSpec(model_name="model_a")
        b = ModelSpec(model_name="model_a", backend="backend_b")
        self.assertTrue(b.fully_contains(a))


if __name__ == '__main__':
    unittest.main()
