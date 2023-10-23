
from .circom import *


"""
    Generator Implementation

    override apply function, several variables to be set:

        1. circom_input
        2. circom_args
        3. circom_output
        4. shape 
"""

class InputGenerator(CircomGenerator):
    def apply(self):
        self.circom_output = self.name

    def fill_circom(self, code: str) -> str:
        if self._visit_flag:
            return code
        self._visit_flag = True

        circom_shape = [str(s) for s in self.shape ]
        circom_shape = ["["+s+"]" for s in circom_shape]
        return inject_signal(code, "signal input {}{};".format(
                self.name, "".join(circom_shape)))


class OutputGenerator(CircomGenerator):
    def apply(self):
        pass

    def fill_circom(self, code: str) -> str:
        if self._visit_flag:
            return code
        self._visit_flag = True

        assert len(self.inputs) == 1
        assert self.shape == self.inputs[0].shape

        for inp in self.inputs:
            code = inp.fill_circom(code)

        circom_shape_lst = ["["+str(s)+"]" for s in self.shape]
        code = inject_signal(code, "signal output {}{};".format(
                    self.name, "".join(circom_shape_lst)))

        circom_shape = "{main}"
        for idx, dim in enumerate(self.shape):
            circom_for = (
                "for (var i{idx} = 0; i{idx} < {dim}; i{idx}++) {brace_left}\n"
                "{main}\n"
                "{brace_right}\n"
            ).format_map(SafeDict(idx=idx, dim=dim))
            circom_shape = circom_shape.format_map(
                    SafeDict(main=circom_for.strip()))

        circom_index = ["[i"+str(i)+"]" \
                for i in range(len(self.shape))]
        circom_assign = "\t{}{} <== {}{};".format(
                self.name, "".join(circom_index),
                self.inputs[0].circom_output, "".join(circom_index),
                )
        circom_shape = circom_shape.format_map(
                SafeDict(main=circom_assign))
        return inject_main(code, circom_shape)

class OperatorGenerator(CircomGenerator):
    def apply(self):
        input_shapes = [inp.shape for inp in self.inputs]
        # check input shape dimensions match operators in cirom circuit operator
        # print(self.comp.input_dims, input_shapes, self.info(), self.comp.input_names)

        assert len(self.comp.input_names) == len(self.inputs)
        # op dim contains 1-dim batch, not support in circom circuits
        for shape in zip(self.comp.input_dims, input_shapes):
            # model input shape dimensions should match cirom circuit operator shape
            assert shape[0] == len(shape[1]), (
                "{}({}) shape dim not matched, "
                "{} vs. {}, maybe apply shape-adaptor pass."
            ).format(self.name, self.comp.op_name,
                    shape[0], len(shape[1]))

        self.circom_inputs = [
                Signal(self, *info) for info in zip(
                    self.comp.input_names, input_shapes) ]

        args = self.arguments()
        # all arguments of circom circuit must be integers.
        assert all([isinstance(a, int) for a in args]), print("bad arg display", [a for a in args], self.info()) #self.info()
        self.circom_args = ", ".join([
            str(s) for s in self.arguments()])

        #  self.circom_output = self.output_name()
        assert len(self.comp.output_names) == 1, "names:{}, dims:{}".format(self.comp.output_names, self.comp.output_dims)
        self.circom_output = "{}.{}".format(self.name, self.comp.output_names[0])

        # check output shape dimensions match
        assert self.comp.output_dims[0] == len(self.shape), self.info()

    def arguments(self):
        raise NotImplementedError(self.comp.op_name)

class ShapeGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.shape ]
class Broadcast3DAxis0SubGenerator(ShapeGenerator):
    pass
class Broadcast3DAxis0AddGenerator(ShapeGenerator):
    pass
class Element1DAddGenerator(ShapeGenerator):
    pass
class Element3DAddGenerator(ShapeGenerator):
    pass
class Element1DSubGenerator(ShapeGenerator):
    pass
class Element1DMulGenerator(ShapeGenerator):
    pass

#  class ElementGenerator(OperatorGenerator):
#      def arguments(self):
#          return [ self.shape[0], ]
#  class ElementAddGenerator(ElementGenerator):
#      pass
#  class ElementSubGenerator(ElementGenerator):
#      pass
#  class ElementMulGenerator(ElementGenerator):
#      pass

# just for test, invalid now
class Conv2D_NCHWGenerator(OperatorGenerator):
    def arguments(self):
        padding = self.attrs["padding"]
        assert all([p == 0 for p in padding])

        filters = self.attrs["channels"]
        kernel_size = self.attrs["kernel_size"]
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        assert kernel_size[0] == kernel_size[1]
        kernel_size = kernel_size[0]
        return [ *self.inputs[0].shape, filters, kernel_size, 1, ]

class Conv2D_CHWGenerator(OperatorGenerator):
    def arguments(self):

        strides = self.attrs["strides"] if len(self.attrs["strides"]) > 0 else [1]
        assert all([s == strides[0] for s in strides])
        padding = self.attrs["padding"]
        assert all([p == 0 for p in padding])
        dilation = self.attrs["dilation"]
        assert all([d == 1 for d in dilation])

        filters = self.attrs["channels"]
        kernel_size = self.attrs["kernel_size"]
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        assert kernel_size[0] == kernel_size[1]
        kernel_size = kernel_size[0]
        return [ *self.inputs[0].shape, filters, kernel_size, strides[0], ]

class MaxPool2DGenerator(OperatorGenerator):
    def arguments(self):
        strides = self.attrs["strides"] if len(self.attrs["strides"]) > 0 else [1]
        assert all([s == strides[0] for s in strides])
        padding = self.attrs["padding"]
        assert all([p == 0 for p in padding])
        dilation = self.attrs["dilation"]
        assert all([d == 1 for d in dilation])
        pool_size = self.attrs["pool_size"]
        assert all([p == pool_size[0] for p in pool_size])

        return [ *self.inputs[0].shape, pool_size[0], strides[0], ]


class Pad2DGenerator(OperatorGenerator):
    def arguments(self):
        pad_value = self.attrs.get("scalar", None)
        if pad_value is None:
            pad_value = self.attrs["pad_value"]
        pad_width = [p for p in self.attrs["padding"]]
        return [ *self.inputs[0].shape, pad_value, *pad_width ]

class BiasAdd1Generator(OperatorGenerator):
    def arguments(self):
        assert len(self.inputs) == 2
        assert len(self.inputs[0].shape) == 1
        assert len(self.inputs[1].shape) == 1
        assert self.inputs[0].shape[0] == self.inputs[1].shape[0]
        return [ *self.inputs[0].shape ]
class BiasAdd2Generator(OperatorGenerator):
    def arguments(self):
        assert len(self.inputs) == 2
        assert len(self.inputs[0].shape) == 2
        assert len(self.inputs[1].shape) == 1
        assert self.inputs[0].shape[0] == self.inputs[1].shape[0]
        return [ *self.inputs[0].shape ]
class BiasAdd3Generator(OperatorGenerator):
    def arguments(self):
        assert len(self.inputs) == 2
        assert len(self.inputs[0].shape) == 3
        assert len(self.inputs[1].shape) == 1
        assert self.inputs[0].shape[0] == self.inputs[1].shape[0]
        return [ *self.inputs[0].shape ]


class Resize2DGenerator(OperatorGenerator):
    def arguments(self):
        method = self.attrs.get("method", "nearest_neighbor")
        assert method == "nearest_neighbor"

        input_shape = self.inputs[0].shape
        scaleX = self.shape[1] / input_shape[1]
        scaleY = self.shape[2] / input_shape[2]
        assert scaleX == scaleY
        assert int(scaleX) == scaleX
        return self.inputs[0].shape + [ int(scaleX), ]


def reshape_validate(shape_one, shape_arr, msg):
        assert len(shape_one) == 1
        total_len = 1
        for s in shape_arr:
            total_len *= s
        assert shape_one[0] == total_len, msg

class ReShapeGenerator(OperatorGenerator):
    def arguments(self):
        reshape_validate(
                self.inputs[0].shape,
                self.shape, self.info())
        return self.shape
class ReShape2DGenerator(ReShapeGenerator):
    pass
class ReShape3DGenerator(ReShapeGenerator):
    pass
class ReShape4DGenerator(ReShapeGenerator):
    pass

class FlattenGenerator(OperatorGenerator):
    def arguments(self):
        reshape_validate(
                self.shape,
                self.inputs[0].shape, self.info())
        return self.inputs[0].attrs["shape"]
class Flatten2DGenerator(FlattenGenerator):
    pass
class Flatten3DGenerator(FlattenGenerator):
    pass
class Flatten4DGenerator(FlattenGenerator):
    pass

class Dense2Generator(OperatorGenerator):
    def arguments(self):
        return self.inputs[1].shape

class ScalarGenerator(OperatorGenerator):
    def arguments(self):
        ishape = self.inputs[0].shape
        assert len(ishape) == 1
        return [ishape[0], self.attrs["scalar"]]
class MulScalarGenerator(ScalarGenerator):
    pass
class MulScalarCHWGenerator(ScalarGenerator):
    def arguments(self):
        i_shape = self.inputs[0].shape
        s_shape = self.inputs[1].shape
        # s_shape[0] is batch, should be 1, then just ignored
        assert len(i_shape) == 3
        assert len(s_shape) == 4
        assert s_shape[0] == 1 and s_shape[2] == 1 and s_shape[3] == 1
        assert i_shape[0] == s_shape[1]
        return [ *i_shape ]
class AddScalarGenerator(ScalarGenerator):
    pass
class SubScalarGenerator(ScalarGenerator):
    pass

#class MulScalarGenerator(OperatorGenerator):
#    def arguments(self):
#        return [ *self.shape,  ]
class RightShiftGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.shape, self.attrs["scalar"]]

class ReLU1DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.shape ]
class ReLU2DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.shape ]
class ReLU3DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.shape ]

class Pass1DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.inputs[0].shape ]
class Pass2DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.inputs[0].shape ]
class Pass3DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.shape ]
class Pass4DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.inputs[0].shape ]
class Pass5DGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.inputs[0].shape ]

class Sum_CHWGenerator(OperatorGenerator):
    def arguments(self):
        keepdims = self.attrs["keepdims"]
        assert(keepdims == None or keepdims == True)
        return [ *self.inputs[0].shape ]

class Sum_CHW_0Generator(OperatorGenerator):
    def arguments(self):
        keepdims = self.attrs["keepdims"]
        assert(keepdims == False)
        return [ *self.inputs[0].shape ]

class Squeeze_CHWGenerator(OperatorGenerator):
    def arguments(self):
        iShape = self.inputs[0].shape
        assert(len(iShape)==3 and iShape[1]==1 and iShape[2]==1)
        return [ *self.inputs[0].shape ]

class Clip1DGenerator(OperatorGenerator):
    def arguments(self):
        assert int(self.attrs["a_min"]) == self.attrs["a_min"]
        assert int(self.attrs["a_max"]) == self.attrs["a_max"]
        return [ self.shape[0],
                int(self.attrs["a_min"]), int(self.attrs["a_max"]) ]

class Clip2DGenerator(OperatorGenerator):
    def arguments(self):
        assert int(self.attrs["a_min"]) == self.attrs["a_min"]
        assert int(self.attrs["a_max"]) == self.attrs["a_max"]
        return [ self.shape[0], self.shape[1],
                int(self.attrs["a_min"]), int(self.attrs["a_max"]) ]

class Clip3DGenerator(OperatorGenerator):
    def arguments(self):
        assert int(self.attrs["a_min"]) == self.attrs["a_min"]
        assert int(self.attrs["a_max"]) == self.attrs["a_max"]
        return [ self.shape[0], self.shape[1], self.shape[2],
                int(self.attrs["a_min"]), int(self.attrs["a_max"]) ]

class TransposeC1C2HWGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==4)
        assert(self.attrs["axes"][1:]==[2,1,3,4]), self.attrs["axes"]
        # only transpose C1 and C2
        return [ *self.inputs[0].shape ]

class TupleGetItem3DGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.inputs[0].shape)==3)
        return [ *self.inputs[0].shape, self.attrs["index"] ]

class Concatenate3DGenerator(OperatorGenerator):
    def arguments(self):
        assert(len(self.shape)==3)
        assert(len(self.inputs)==2)
        assert all([self.inputs[0].shape[1] == self.inputs[1].shape[1], self.inputs[0].shape[2] == self.inputs[1].shape[2]])
        return [ self.inputs[0].shape[0], *self.inputs[1].shape ]
