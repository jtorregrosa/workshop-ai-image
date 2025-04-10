class ModelManager:
    def __init__(self, loaders: dict[str, callable]):
        self._loaders = loaders
        self._active_key = None
        self._active_model = None

    def get_model(self, key: str):
        if key == self._active_key:
            return self._active_model  # Already loaded

        if key not in self._loaders:
            raise ValueError(f"Unknown model key: {key}")

        # Unload current model
        self._unload_active_model()

        # Load new model
        self._active_model = self._loaders[key]()
        self._active_key = key
        return self._active_model

    def _unload_active_model(self):
        if self._active_model is not None:
            # Optional: call cleanup hooks if models require it
            del self._active_model
            self._active_model = None
            self._active_key = None

    def unload(self):
        self._unload_active_model()

    def current_key(self):
        return self._active_key

    def is_loaded(self, key: str) -> bool:
        return key == self._active_key