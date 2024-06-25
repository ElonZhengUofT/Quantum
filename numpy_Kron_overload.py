import numpy as np

class Matrix(np.ndarray):
    def __new__(cls, input_array):
        # 创建一个 ndarray 实例，并将其视图转换为 Matrix
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        # 如果需要，可以在这里进行额外的初始化
        if obj is None: return

    def __pow__(self, other):
        if isinstance(other, Matrix):
            return Matrix(np.kron(self, other))
        return NotImplemented

    def __repr__(self):
        return f'Matrix({super().__repr__()})'

    def __round__(self, n=None):
        return Matrix(np.round(self, n))