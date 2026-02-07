import numpy as np


class Tensor:
    def __init__(self, input, *, device=None, dtype="float32", requires_grad=True):
        """
        Create a new tensor.

        Args:
            data: Array-like data (list, numpy array, or another Tensor)
            device: Device placement (currently ignored, CPU only)
            dtype: Data type for the array
            requires_grad: Whether to track gradients for this tensor
        """
        # Step 1: Normalize the input to a numpy array
        # We need to handle three cases and end up with `data` being a numpy array

        if isinstance(input, Tensor):
            # Case 1: Unwrap the Tensor
            input = input.data

        elif isinstance(input, np.ndarray):
            # Case 2: Already a numpy array
            pass

        else:
            # Case 3: Something else (list, scalar, etc.)
            input = np.array(input)

        # Store with the correct dtype
        self.data = input.astype(dtype)

        # Initialize remaining attributes
        self.requires_grad = requires_grad
        self.grad = None
        self._device = device
        self._op = None
        self._inputs = []

    @property
    def shape(self):
        """Shape of the tensor."""
        return self.data.shape

    @property
    def dtype(self):
        """Data type of the tensor."""
        return self.data.dtype

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.data.ndim

    @property
    def size(self):
        """Total number of elements."""
        return self.data.size

    @property
    def device(self):
        """Device where tensor lives."""
        return self._device

    def numpy(self):
        """
        Return the data as a NumPy array (detached from the computation graph).
        This returns a copy, so modifying the result will not affect
        the tensor's data.

        Examples:
            >>> x = Tensor([1, 2, 3])
            >>> y = x + 1   # y is still a Tensor, part of the graph
            >>> z = x.numpy() + 1  # z is a NumPy array, not part of the graph

        Returns:
            np.ndarray: A copy of the tensor's data as a NumPy array.
        """
        return self.data.copy()

    def detach(self):
        """
        Creates a new Tensor with same data but no gradient tracking.
        Useful when you want to use values without building
        computation graph.

        Returns:
            Tensor: New tensor with requires_grad=False

        Example:
            >>> x = Tensor([1, 2, 3], requires_grad=True)
            >>> y = x.detach()  # y doesn't track gradients
            >>> z = y * 2       # This operation won't be in graph
        """
        return Tensor(self.data, requires_grad=False)
