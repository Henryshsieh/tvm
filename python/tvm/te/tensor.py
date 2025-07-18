# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tensor class for computation declaration."""
# pylint: disable=invalid-name
import tvm.ffi

from tvm.runtime import Object, ObjectGeneric
from tvm.tir import expr as _expr, DataProducer

from . import _ffi_api


class TensorSlice(ObjectGeneric, _expr.ExprOp):
    """Auxiliary data structure for enable slicing syntax from tensor."""

    def __init__(self, tensor, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        self.tensor = tensor
        self.indices = indices

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        return TensorSlice(self.tensor, self.indices + indices)

    def asobject(self):
        """Convert slice to object."""
        return self.tensor.__call__(*self.indices)

    @property
    def dtype(self):
        """Data content of the tensor."""
        return self.tensor.dtype


@tvm.ffi.register_object("te.Tensor")
class Tensor(DataProducer, _expr.ExprOp):
    """Tensor object, to construct, see function.Tensor"""

    def __call__(self, *indices):
        ndim = self.ndim
        if len(indices) != ndim:
            raise ValueError(
                f"Need to provide {ndim} index in tensor but {len(indices)} was provided"
            )
        return _expr.ProducerLoad(self, indices)

    def __getitem__(self, indices):
        return TensorSlice(self, indices)

    def __hash__(self):
        return _ffi_api.TensorHash(self)

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            if isinstance(other, _expr.ExprOp):
                return _expr.EqualOp(self, other)
            return False
        if self.ndim == 0 and other.ndim == 0:
            raise ValueError(
                "Equal == comparison among rank-0 tensor is ambiguous, "
                "use Tensor.equal for content expression equvalence, "
                "use Tensor.same_as for exact reference comparison"
            )
        return _ffi_api.TensorEqual(self, other)

    @property
    def ndim(self):
        """Dimension of the tensor."""
        return len(self.shape)

    @property
    def axis(self):
        """Axis of the tensor."""
        return self.__getattr__("axis")

    @property
    def op(self):
        """The corressponding :py:class:`Operation`."""
        return self.__getattr__("op")

    @property
    def value_index(self):
        """The output value index the tensor corresponds to."""
        return self.__getattr__("value_index")

    @property
    def shape(self):
        """The output shape of the tensor."""
        return self.__getattr__("shape")

    @property
    def name(self):
        op = self.op
        if op.num_outputs == 1:
            return op.name
        return f"{op.name}.v{self.value_index}"


@tvm.ffi.register_object("te.Operation")
class Operation(Object):
    """Represent an operation that generates a tensor"""

    def output(self, index):
        """Get the index-th output of the operation

        Parameters
        ----------
        index : int
            The index size.

        Returns
        -------
        out : Tensor
            The i-th output.
        """
        return _ffi_api.OpGetOutput(self, index)

    @property
    def num_outputs(self):
        """Number of outputs from this op."""
        return _ffi_api.OpNumOutputs(self)

    @property
    def input_tensors(self):
        """List of input tensors to this op."""
        return _ffi_api.OpInputTensors(self)


@tvm.ffi.register_object("te.PlaceholderOp")
class PlaceholderOp(Operation):
    """Placeholder operation."""


@tvm.ffi.register_object("te.BaseComputeOp")
class BaseComputeOp(Operation):
    """Compute operation."""

    @property
    def axis(self):
        """Represent the IterVar axis, defined when it is a ComputeOp"""
        return self.__getattr__("axis")

    @property
    def reduce_axis(self):
        """Represent axis of reductions, only defined when it is a ComputeOp"""
        return self.__getattr__("reduce_axis")


@tvm.ffi.register_object("te.ComputeOp")
class ComputeOp(BaseComputeOp):
    """Scalar operation."""


@tvm.ffi.register_object("te.ScanOp")
class ScanOp(Operation):
    """Scan operation."""

    @property
    def scan_axis(self):
        """Represent the scan axis, only defined when it is a ScanOp"""
        return self.__getattr__("scan_axis")


@tvm.ffi.register_object("te.ExternOp")
class ExternOp(Operation):
    """External operation."""
