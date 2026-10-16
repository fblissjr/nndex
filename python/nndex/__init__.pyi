from typing import Any, Literal, TypedDict, overload
import numpy as np
from numpy.typing import NDArray

class GpuDeviceInfoDict(TypedDict):
    index: int
    name: str
    driver: str
    driver_info: str
    backend: str
    device_type: str

def list_gpu_devices() -> list[GpuDeviceInfoDict]: ...

class NNdex:
    def __init__(
        self,
        data: Any,
        normalized: bool = False,
        approx: bool = False,
        backend: Literal["cpu", "gpu"] = "cpu",
        enable_cache: bool = True,
        gpu_device_index: int | None = None,
    ) -> None: ...

    @classmethod
    def from_file(
        cls,
        path: str,
        *,
        key: str | None = None,
        normalized: bool = False,
        approx: bool = False,
        backend: Literal["cpu", "gpu"] = "cpu",
        enable_cache: bool = True,
        gpu_device_index: int | None = None,
    ) -> NNdex: ...

    @property
    def backend(self) -> Literal["cpu", "gpu"]: ...

    @property
    def rows(self) -> int: ...

    @property
    def dims(self) -> int: ...

    @property
    def gpu_info(self) -> GpuDeviceInfoDict | None: ...

    @overload
    def search(
        self,
        query: Any,
        k: int = 10,
        dataframe: None = None,
    ) -> tuple[NDArray[np.uint64], NDArray[np.float32]]: ...

    @overload
    def search(
        self,
        query: Any,
        k: int = 10,
        *,
        dataframe: Any,
    ) -> Any: ...
