import abc
import importlib
import inspect
import json
import os
import logging
import logging.config

from typing import Dict, List, Tuple, Any, Type

import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
with open(os.path.join(project_root, "logging.yaml")) as f:
    conf = yaml.safe_load(f)
    log_fn = conf["handlers"]["file_handler"]["filename"]
    log_fn = os.path.join(project_root, log_fn)
    conf["handlers"]["file_handler"]["filename"] = log_fn
    logging.config.dictConfig(conf)


def get_logger(name):
    return logging.getLogger(name)


# Load backend dynamically from "backends" sibling directory
# Note: The backends might use get_logger (circular import)
def load_credentials(backend, file_name="key.json") -> Dict:
    key_file = os.path.join(project_root, file_name)
    with open(key_file) as f:
        creds = json.load(f)
    assert backend in creds, f"No '{backend}' in {file_name}. See README."
    assert "api_key" in creds[backend], f"No 'api_key' in {file_name}. See README."
    return creds


class ModelSpec:
    PROGRAMMATIC_SPECS = ["mock", "dry_run", "programmatic", "custom", "_slurk_response"]
    HUMAN_SPECS = ["human", "terminal"]

    def __init__(self, model_name: str, backend: str = None, model_id: str = None,
                 temperature: float = None, opts: Dict = None):
        super().__init__()
        self.model_name = model_name  # mandatory
        self.backend = backend
        self.model_id = model_id  # model name as specified in the backend (backends can fall back to model_name)
        self.temperature = temperature
        self.opts = dict() if opts is None else opts

    def update(self, other: "ModelSpec"):
        # keep ours if already specified
        if not self.has_backend():
            self.backend = other.backend
        if not self.has_model_id():
            self.model_id = other.model_id
        if not self.has_temperature():
            self.temperature = other.temperature
        other_opts = other.opts.copy()
        other_opts.update(self.opts)  # keep ours
        self.opts = other_opts

    def matches(self, other: "ModelSpec") -> bool:
        if self.model_name != other.model_name:
            return False

        if self.has_backend():  # compare both
            if self.backend != other.backend:
                return False

        return True

    def has_temperature(self):
        return self.temperature is not None

    def has_backend(self):
        return self.backend is not None

    def has_model_id(self):
        return self.model_id is not None

    @classmethod
    def from_name(cls, model_name: str):
        if model_name is None:
            raise ValueError(f"Cannot create ModelSpec because model_name is None (but required)")
        return cls(model_name=model_name)

    @classmethod
    def from_dict(cls, spec: Dict):
        if "model_name" not in spec:
            raise ValueError(f"Missing 'model_name' in model spec: {spec}")
        model_name = spec.pop("model_name")
        model_spec = cls.from_name(model_name)
        model_spec.backend = spec.pop("backend", None)
        model_spec.model_id = spec.pop("model_id", None)
        model_spec.temperature = spec.pop("temperature", None)
        model_spec.opts = spec
        return model_spec

    def is_programmatic(self):
        return self.model_name in ModelSpec.PROGRAMMATIC_SPECS

    def is_human(self):
        return self.model_name in ModelSpec.HUMAN_SPECS


class Backend(abc.ABC):

    def __init__(self, model_spec: ModelSpec):
        self.model_spec = model_spec
        if not model_spec.has_temperature():
            model_spec.temperature = 0.0
        if not model_spec.has_model_id():
            model_spec.model_id = model_spec.model_name  # fallback to model name

    @abc.abstractmethod
    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        """Put prompt in model-specific format and get its response.

        Args:
            messages (List[Dict]): The dialogue context represented as a list
                of turns. Entry element is a dictionary containing one key
                "role", whose value is either "user" or "assistant", and one
                key "content", whose value is the message as a string.
            model (str): the name of the model

        Returns:
            Tuple[Any, Any, str]: The first element is the prompt object as 
            passed to the LLM (i.e. after any model-specific manipulation).
            Return the full prompt object, not only the message string.

            The second element is the response object as gotten from the model,
            before any manipulation. Return the full prompt object, not only 
            the message string.

            These must be returned just to be logged by the GM for later inspection.

            The third element is the response text, i.e. only the actual message
            generated by the model as a string (after any needed manipulation,
            like .strip() or excluding the input prompt).
        """
        pass

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.__class__.__name__}({self.model_spec.model_name})"


class ProgrammaticBackend(Backend):

    def __init__(self, model_spec=ModelSpec("programmatic")):
        super().__init__(model_spec)

    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        raise NotImplementedError("This should never be called but is handled in Player")


class HumanBackend(Backend):

    def __init__(self, model_spec=ModelSpec("human")):
        super().__init__(model_spec)

    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        raise NotImplementedError("This should never be called but is handled in Player")


def is_backend(obj):
    if inspect.isclass(obj) and issubclass(obj, Backend):
        return True
    return False


_backend_registry: Dict[str, Type] = dict()  # we store references to the class constructor
_model_registry: List[ModelSpec] = list()  # we store model specs so that users might use model_name for lookup


def init_model_registry(_model_registry_path: str = None):
    if not _model_registry_path:
        _model_registry_path = os.path.join(project_root, "backends", "model_registry.json")
    if not os.path.isfile(_model_registry_path):
        raise FileNotFoundError(f"The file model registry at '{_model_registry_path}' does not exist. "
                                f"Create model registry as a model_registry.json file and try again.")
    with open(_model_registry_path) as f:
        _model_listing = json.load(f)
        for _model_entry in _model_listing:
            _model_spec: ModelSpec = ModelSpec.from_dict(_model_entry)
            if not _model_spec.has_backend():
                raise ValueError(
                    f"Missing backend definition in model registry for model_name='{_model_spec.model_name}'. "
                    f"Check or update the backends/model_registry.json and try again."
                    f"A minimal entry is {{'model_name':<name>,'backend':<backend>}}.")
            _model_registry.append(_model_spec)


def _register_backend(backend_name: str):
    """
    Dynamically loads the Backend in the file with name <backend_name>_api.py into the _backend_registry.
    Raises an exception if no such file exists or the Backend class could not be found.

    :param backen_name: the prefix of the <backend_name>_api.py file
    """
    backends_root = os.path.join(project_root, "backends")
    backend_module = f"{backend_name}_api"
    backend_path = os.path.join(backends_root, f"{backend_module}.py")
    if not os.path.isfile(backend_path):
        raise FileNotFoundError(f"The file '{backend_path}' does not exist. "
                                f"Create such a backend file or check the backend_name '{backend_name}'.")
    module = importlib.import_module(f"backends.{backend_module}")
    backend_subclasses = inspect.getmembers(module, predicate=is_backend)
    if len(backend_subclasses) == 0:
        raise LookupError(f"There is no Backend defined in {backend_module}. "
                          f"Create such a class and try again or check the backend_name '{backend_name}'.")
    if len(backend_subclasses) > 1:
        raise LookupError(f"There is more than one Backend defined in {backend_module}.")
    _, backend_cls = backend_subclasses[0]
    _backend_registry[backend_name] = backend_cls
    return backend_cls


def _load_backend_for(model_spec: ModelSpec):
    backend_name = model_spec.backend
    if backend_name not in _backend_registry:
        _register_backend(backend_name)
    backend_cls = _backend_registry[backend_name]
    return backend_cls(model_spec)


def get_backend_for(model_spec: ModelSpec) -> Backend:
    """
    :param model_spec: the model spec for which a supporting backend has to be found
    :return: the backend registered that supports the model
    """
    if model_spec.is_human():
        return HumanBackend(model_spec)
    if model_spec.is_programmatic():
        return ProgrammaticBackend(model_spec)
    for registered_spec in _model_registry:
        if model_spec.matches(registered_spec):
            model_spec.update(registered_spec)
    if not model_spec.has_backend():
        raise ValueError(f"Model spec requires backend, but no entry in model registry "
                         f"for model_name='{model_spec.model_name}'. "
                         f"Check or update the backends/model_registry.json and try again. "
                         f"A minimal entry is {{'model_name':<name>,'backend':<backend>}}.")
    backend = _load_backend_for(model_spec)
    return backend
