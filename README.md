<img src=https://raw.githubusercontent.com/zk-ml/linear-a-site/main/logo/linear-a-logo.png width=64/> tachikoma: neural network inference standard for zero-knowledge-proof systems


---------------

tachikoma defines how a neural network's inference process should be serialized into a graph of operator computational traces, each of which containing the input, expected output, relevant metadata (including parameters), and an identifier relating back to the original operator in TVM's intermediate representation.

---------------

We are actively working on consolidating the standards into a stable form and release relevant artifacts, as well as forming a committee and organizing regular meetings. If you are interested in this effort, please reach out!

---------------

in addition, tachikoma's TVM fork is useful for:
- converting a floating-point neural network or a framework-prequantized model into an integer-only form
- generating a computational trace binary respecting the tachikoma standard.
- as a proof of concept, how tachikoma can be used in ZKP systems. We will be implementing a simple graph runtime on top of the tachikoma standard, as well as a circuit builder in ZEXE. The code will be available here: https://github.com/zk-ml/tachikoma-poc-runtime

---------------

## MRT + ZKML

*Developed by CortexLabs Team. All models are derived from the gluoncv model zoo.*

Current MRT has supported most CNN classification models and limited detection models. More details are listed as following:

### Classification Model

*The main test code is located at tests/models/classification/test.main.py.*

- "resnet18"                 # passed

  Iteration: 19 | from_expr: Top1/5: 77.50%,94.69% | sim: Top1/5: 77.50%,94.69% | clip: Top1/5: 77.81%,94.69% | round: Top1/5: 75.62%,94.06% | quantized: Top1/5: 75.31%,94.06% |

- "mobilenet_v2"             # passed
- "alexnet"                  # passed

  Iteration: 19 | from_expr: Top1/5: 66.88%,88.44% | sim: Top1/5: 66.88%,88.44% | clip: Top1/5: 67.19%,88.44% | round: Top1/5: 66.56%,89.06% | quantized: Top1/5: 66.56%,89.06% |

- "densenet121"              # passed

  Iteration:  19 | from_expr: Top1/5: 83.75%,96.56% | sim: Top1/5: 83.75%,96.56% | clip: Top1/5: 83.75%,96.88% | round: Top1/5: 51.25%,81.25% | quantized: Top1/5: 50.31%,81.25% |

- "squeezenet1_0"            # passed

  Iteration:  19 | from_expr: Top1/5: 70.31%,91.56% | sim: Top1/5: 70.31%,91.56% | clip: Top1/5: 70.31%,91.56% | round: Top1/5: 59.06%,85.94% | quantized: Top1/5: 59.38%,85.94% |

- "vgg11"                    # passed

  Iteration:  19 | from_expr: Top1/5: 79.06%,95.00% | sim: Top1/5: 79.06%,95.00% | clip: Top1/5: 79.06%,95.00% | round: Top1/5: 77.50%,95.94% | quantized: Top1/5: 77.50%,95.94% |

- "shufflenet_v2_x0_5"       # passed

  Iteration:  19 | from_expr: Top1/5: 73.12%,90.00% | sim: Top1/5: 73.12%,90.00% | clip: Top1/5: 73.75%,89.69% | round: Top1/5: 0.94%,4.38% | quantized: Top1/5: 0.94%,4.38% |

### Detection Models

*The main test code is located at tests/models/detection/test.mxnet.py.*

- "mxnet_ssd_512_resnet50_v1_voc"     # passed
- "yolo3_darknet53_voc"               # passed
