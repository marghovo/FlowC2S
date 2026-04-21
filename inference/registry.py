PIPELINE_REGISTRY = {}

def register(name):
    def decorator(cls):
        PIPELINE_REGISTRY[name] = cls
        return cls
    return decorator
