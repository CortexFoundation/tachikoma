""" MRT operator names """

VAR = "var"

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

# =========== NON-CALC ops ===============
TUPLE = "Tuple"
TUPLE_GET_ITEM = "TupleGetItem"

REPEAT = "repeat"
SQUEEZE = "squeeze"
FLATTEN = "flatten"
BATCH_FLATTEN = "nn.batch_flatten"
RESHAPE = "reshape"
CONCAT = "concatenate"
SPLIT = "split"
TRANSPOSE = "transpose"

WHERE = "where"
GREATER = "greater"
STRIDED_SLICE = "strided_slice"
GET_VALID_COUNT = "vision.get_valid_counts"
NON_MAX_SUPRESSION = "vision.non_max_suppression"

CLIP = "clip"
CEIL = "ceil"
RIGHT_SHIFT = "right_shift"
# AS_TYPE = "astype"
CAST = "cast"

# ======= binary ops =============

ADD = "add"
SUB = "sub"
MUL = "multiply"

# ======= auto generate op =========
ARANGE = "arange"
ZEROS_LIKE = "zeros_like"

# ======= control flow op ===========
IF = "if"
ARGWHERE = "argwhere"

# ======= mrt requant op ==========
REQUANT = "mrt.requant"
PCLIP = "mrt.pclip"
""" precision clip """
RS_PCLIP = "mrt.rs_pclip"
""" right shift precision clip """

