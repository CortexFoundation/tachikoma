pragma circom 2.1.0;

template Pass1D (i1) {
    signal input in[i1];
    signal output out[i1];
    for (var i = 0; i < i1; i++) {
	out[i] <== in[i];
    }
}

template Pass2D (i1, i2) {
    signal input in[i1][i2];
    signal output out[i1][i2];
    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
	    out[i][j] <== in[i][j];
        }
    }
}

template Pass3D (i1, i2, i3) {
    signal input in[i1][i2][i3];
    signal output out[i1][i2][i3];
    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                out[i][j][k] <== in[i][j][k];
            }
        }
    }
}

template Pass4D (i1, i2, i3, i4) {
    signal input in[i1][i2][i3][i4];
    signal output out[i1][i2][i3][i4];
    for (var x1 = 0; x1 < i1; x1++) {
        for (var x2 = 0; x2 < i2; x2++) {
            for (var x3 = 0; x3 < i3; x3++) {
                for (var x4 = 0; x4 < i4; x4++) {
                    out[x1][x2][x3][x4] <== in[x1][x2][x3][x4];
                }
            }
        }
    }
}

template Pass5D (i1, i2, i3, i4, i5) {
    signal input in[i1][i2][i3][i4][i5];
    signal output out[i1][i2][i3][i4][i5];
    for (var x1 = 0; x1 < i1; x1++) {
        for (var x2 = 0; x2 < i2; x2++) {
            for (var x3 = 0; x3 < i3; x3++) {
                for (var x4 = 0; x4 < i4; x4++) {
                    for (var x5 = 0; x5 < i5; x5++) {
                        out[x1][x2][x3][x4][x5] <== in[x1][x2][x3][x4][x5];
                    }
                }
            }
        }
    }
}

template TransposeHWC (i1, i2, i3) {
    signal input in[i1][i2][i3];
    // 2,3,1
    signal output out[i2][i3][i1];

    for (var x1 = 0; x1 < i1; x1++) {
        for (var x2 = 0; x2 < i2; x2++) {
            for (var x3 = 0; x3 < i3; x3++) {
                // 2,3,1
                out[x2][x3][x1] <== in[x1][x2][x3];
            }
        }
    }

}

template TransposeC2C1HW (i1, i2, i3, i4) {
    signal input in[i1][i2][i3][i4];
    // 2,1,3,4
    signal output out[i2][i1][i3][i4];

    for (var x1 = 0; x1 < i1; x1++) {
        for (var x2 = 0; x2 < i2; x2++) {
            for (var x3 = 0; x3 < i3; x3++) {
                for (var x4 = 0; x4 < i4; x4++) {
                    // 2,1,3,4
                    out[x2][x1][x3][x4] <== in[x1][x2][x3][x4];
                }
            }
        }
    }

}

template Split3D (i1, i2, i3) {
    // deprecated, should pass
    signal input in[i1][i2][i3];
    signal output out1[i1\2][i2][i3];
    signal output out2[i1\2][i2][i3];
}

template TupleGetItem3D (i1, i2, i3, index) {
    signal input in[i1][i2][i3];
    signal output out[i1\2][i2][i3];

    if (index == 0) {
        for (var i = 0; i < i1\2; i++) {
    	    for (var j = 0; j < i2; j++) {
    	        for (var k = 0; k < i3; k++) {
	            out[i][j][k] <== in[i][j][k];
                }
            }
        }
    } else {
        for (var i = i1\2; i < i1; i++) {
    	    for (var j = 0; j < i2; j++) {
    	        for (var k = 0; k < i3; k++) {
                    out[i-i1\2][j][k] <== in[i][j][k];
                }
            }
        }

    }
}

