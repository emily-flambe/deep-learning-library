import numpy as np


class Tensor:
    """A tensor class for deep learning with automatic differentiation support."""

    def __init__(self, data, *, device=None, dtype="float32", requires_grad=True):
        """
        Initialize a Tensor.

        Args:
            data: Input data - can be a list, scalar, NumPy ndarray, or another Tensor
            device: Device placement (currently only "cpu" supported)
            dtype: Data type (default "float32")
            requires_grad: Whether to track gradients (default True)
        """
        # Normalize data to NumPy array
        if isinstance(data, Tensor):
            data = data.data.copy()
        elif isinstance(data, np.ndarray):
            data = data.copy()
        else:
            data = np.array(data)

        self.data = data.astype(dtype)
        self._dtype = dtype
        self.grad = None
        self.requires_grad = requires_grad
        self._op = None  # Operation that created this tensor
        self._inputs = []  # Input tensors used to create this one
        self._device = device if device is not None else "cpu"

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __str__(self):
        return str(self.data)

    @property
    def shape(self):
        """Returns the tensor dimensions."""
        return self.data.shape

    @property
    def dtype(self):
        """Returns the data type."""
        return self._dtype

    @property
    def ndim(self):
        """Returns the number of dimensions."""
        return self.data.ndim

    @property
    def size(self):
        """Returns the total number of elements."""
        return self.data.size

    @property
    def device(self):
        """Returns the device location."""
        return self._device

    def numpy(self):
        """Returns a copy of the internal NumPy array, detached from computation graph."""
        return self.data.copy()

    def detach(self):
        """Creates a new Tensor with the same data but requires_grad=False."""
        return Tensor(self.data, device=self._device, dtype=self._dtype, requires_grad=False)

    def backward(self):
        """Compute gradients via backpropagation. (Stub for future implementation)"""
        pass
