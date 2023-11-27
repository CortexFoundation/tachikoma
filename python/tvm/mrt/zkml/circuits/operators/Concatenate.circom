pragma circom 2.1.0;

template Concatenate1D (i1_0, i1_1) {
    signal input in0[i1_0];
    signal input in1[i1_1];
    signal output out[i1_0 + i1_1];

    for (var i = 0; i < i1_0; i++) {
	out[i] <== in0[i];
    }
    for (var i = 0; i < i1_1; i++) {
	out[i+i1_0] <== in1[i];
    }
}

template Concatenate2D (i1_0, i1_1, i2) {
    signal input in0[i1_0][i2];
    signal input in1[i1_1][i2];
    signal output out[i1_0 + i1_1][i2];

    for (var j = 0; j < i2; j++) {
            for (var i = 0; i < i1_0; i++) {
                out[i][j] <== in0[i][j];
            }
            for (var i = 0; i < i1_1; i++) {
                out[i+i1_0][j] <== in1[i][j];
            }
        }
    }
}

template Concatenate3D (i1_0, i1_1, i2, i3) {
    signal input in0[i1_0][i2][i3];
    signal input in1[i1_1][i2][i3];
    signal output out[i1_0 + i1_1][i2][i3];

    for (var j = 0; j < i2; j++) {
        for (var k = 0; k < i3; k++) {
            for (var i = 0; i < i1_0; i++) {
                out[i][j][k] <== in0[i][j][k];
            }
            for (var i = 0; i < i1_1; i++) {
                out[i+i1_0][j][k] <== in1[i][j][k];
            }
        }
    }
}

