from ._nndex import NNdex

try:
    from ._nndex import list_gpu_devices
except ImportError:
    def list_gpu_devices() -> list[dict[str, object]]:
        return []

__all__ = ["NNdex", "list_gpu_devices"]

