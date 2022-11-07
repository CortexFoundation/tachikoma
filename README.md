<img src=https://raw.githubusercontent.com/zk-ml/linear-a-site/main/logo/linear-a-logo.png width=64/> tachikoma: neural network inference standard for arithmetic circuits


---------------

tachikoma defines how a neural network's inference process should be serialized into a graph of operator computational traces, each of which containing the input, expected output, relevant metadata (including parameters), and an identifier relating back to the original operator in TVM's intermediate representation.

---------------

We are actively working consolidating the standards into a stable form and release relevant artifacts, as well as forming a committee and organizing regular meetings. If you are interested in this effort, please reach out!

---------------

in addition, tachikoma's TVM fork is useful for:
- converting a floating-point neural network or a framework-prequantized model into an integer-only form
- generating a computational trace binary respecting the tachikoma standard.
- as a proof of concept, how tachikoma can be used in ZKP systems. We will be implementing a simple graph runtime on top of the tachikoma standard, as well as a circuit builder in ZEXE. The code will be available here: https://github.com/zk-ml/tachikoma-poc-runtime