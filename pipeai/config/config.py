# pylint: disable=raise-missing-from

class Config(dict):
    """A recursive dictionary with attribute-style access."""

    def __init__(self, d=None, **kwargs):
        super().__init__()
        d = d or {}
        d.update(kwargs)
        for k, v in d.items():
            self[k] = v

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __setitem__(self, key, value):
        # dict -> Config
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        elif isinstance(value, (list, tuple)):
            value = type(value)(
                Config(v) if isinstance(v, dict) and not isinstance(v, Config) else v
                for v in value
            )
        super().__setitem__(key, value)
        if isinstance(key, str) and key.isidentifier():
            super().__setattr__(key, value)

    def __getitem__(self, key):
        # cfg["a.b.c"]
        if isinstance(key, str) and "." in key:
            keys = key.split(".")
        else:
            keys = [key]
        current = self
        for k in keys:
            current = dict.__getitem__(current, k)
        return current

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default

    def has(self, key):
        return self.get(key) is not None

    def update(self, other=None, **kwargs):
        other = other or {}
        other.update(kwargs)
        for k, v in other.items():
            self[k] = v

    def pop(self, key, default=None):
        if hasattr(self, key):
            delattr(self, key)
        return super().pop(key, default)

    def to_dict(self):
        """Convert recursively to a standard dict."""
        result = {}
        for k, v in self.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            elif isinstance(v, list):
                result[k] = [i.to_dict() if isinstance(i, Config) else i for i in v]
            elif isinstance(v, tuple):
                result[k] = tuple(i.to_dict() if isinstance(i, Config) else i for i in v)
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, d):
        return cls(d)
