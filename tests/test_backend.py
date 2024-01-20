import unittest

from backends import get_backend_for, ModelSpec, init_model_registry


class BackendTestCase(unittest.TestCase):
    def test_get_backend_for_model1(self):
        init_model_registry("test-registry.json")
        backend = get_backend_for(ModelSpec(model_name="model1"))
        assert backend.model_spec.backend == "huggingface_local"

    def test_get_backend_for_model2(self):
        init_model_registry("test-registry.json")
        backend = get_backend_for(ModelSpec(model_name="model2"))
        assert backend.model_spec.backend == "huggingface_local"

    def test_get_backend_for_model1_other(self):
        init_model_registry("test-registry.json")
        backend = get_backend_for(ModelSpec(model_name="model1", backend="openai"))
        assert backend.model_spec.backend == "openai"


if __name__ == '__main__':
    unittest.main()
