""" MRT operator names """

VAR = "var"

TUPLE = "Tuple"
TUPLE_GET_ITEM = "TupleGetItem"

CONV2D = "nn.conv2d"
DENSE = "nn.dense"
BATCH_NORM = "nn.batch_norm"
BIAS_ADD = "nn.bias_add"
RELU = "nn.relu"
GLOBAL_AVG_POOL2D = "nn.adaptive_avg_pool2d"
MAX_POOL2D = "nn.max_pool2d"

SQUEEZE = "squeeze"
RESHAPE = "reshape"
SUM = "sum"

ADD = "add"
SUB = "sub"
MUL = "multiply"

# ======= mrt requant op ==========
REQUANT = "mrt.requant"
PCLIP = "mrt.pclip"
""" precision clip """
RS_PCLIP = "mrt.rs_pclip"
""" right shift precision clip """

