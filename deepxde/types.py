import deepxde as dde
from .backend import backend_name
from typing import Callable, List, Optional, Tuple, Union, Any, Sequence, Type, NewType

import builtins

if backend_name == "tensorflow":
    import tensorflow as tf # pylint: disable=import-error
    from .nn.tensorflow import NN
    #TODO
    Tensor = tf.Tensor
    TensorOrTensors = Union[tf.Tensor, Sequence[tf.Tensor]]
    Optimizer = tf.keras.optimizers.Optimizer
    
elif backend_name == "tensorflow.compat.v1":
    import tensorflow.compat.v1 as tf # pylint: disable=import-error
    from .nn.tensorflow_compat_v1 import NN
    #TODO
    Tensor = tf.Tensor
    TensorOrTensors = Union[tf.Tensor, Sequence[tf.Tensor]]
    Optimizer = tf.keras.optimizers.Optimizer

elif backend_name == "pytorch":
    import torch # pylint: disable=import-error
    from .nn.pytorch import NN
    #TODO
    Tensor = torch.Tensor
    TensorOrTensors = Union[torch.Tensor, Sequence[torch.Tensor]]
    Optimizer = torch.optim.Optimizer

elif backend_name == "jax":
    import jax # pylint: disable=import-error
    import jax.numpy as jnp
    from .nn.jax import NN
    #TODO
    Tensor = jnp.ndarray
    TensorOrTensors = Union[jnp.ndarray, Sequence[jnp.ndarray]]
    Optimizer = jax.experimental.optimizers.Optimizer

elif backend_name == "paddle":
    import paddle # pylint: disable=import-error
    from .nn.paddle import NN
    #TODO
    Tensor = paddle.Tensor
    TensorOrTensors = Union[paddle.Tensor, Sequence[paddle.Tensor]]
    Optimizer = paddle.optimizer.Optimizer

else:
    raise RuntimeError("Unknown backend: {}".format(backend_name))


_NN = NN
_Tensor = Tensor
_TensorOrTensors = TensorOrTensors
_Optimizer = Optimizer

_int = builtins.int
_float = builtins.float
_bool = builtins.bool

Number = Union[builtins.int, builtins.float, builtins.bool]