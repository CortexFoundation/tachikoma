""" MRT operator names """

VAR = "var"

TUPLE = "Tuple"
TUPLE_GET_ITEM = "TupleGetItem"

DROP_OUT = "nn.dropout"
CONV2D = "nn.conv2d"
DENSE = "nn.dense"
BATCH_NORM = "nn.batch_norm"
BIAS_ADD = "nn.bias_add"
RELU = "nn.relu"
ADAPTIVE_AVG_POOL2D = "nn.adaptive_avg_pool2d"
AVG_POOL2D = "nn.avg_pool2d"
MAX_POOL2D = "nn.max_pool2d"

SOFTMAX = "nn.softmax"
LOG_SOFTMAX = "nn.log_softmax"

SUM = "sum"
MEAN = "mean"
REPEAT = "repeat"
SQUEEZE = "squeeze"
# FLATTEN = "flatten"
RESHAPE = "reshape"
CONCAT = "concatenate"
SPLIT = "split"
TRANSPOSE = "transpose"

CLIP = "clip"
CEIL = "ceil"
RIGHT_SHIFT = "right_shift"
# AS_TYPE = "astype"
CAST = "cast"

ADD = "add"
SUB = "sub"
MUL = "multiply"

# ======= auto generate op =========
ARANGE = "arange"

# ======= control flow op ===========
IF = "if"
ARGWHERE = "argwhere"

# ======= mrt requant op ==========
REQUANT = "mrt.requant"
PCLIP = "mrt.pclip"
""" precision clip """
RS_PCLIP = "mrt.rs_pclip"
""" right shift precision clip """

