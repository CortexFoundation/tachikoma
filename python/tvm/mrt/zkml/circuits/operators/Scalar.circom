pragma circom 2.0.3;

template MulScalar(iShape, sc) {
    signal input in[iShape];
    signal output out[iShape];

    for (var i=0; i < iShape; i++) {
        out[i] <== in[i] * sc;
    }
}

// scalar depends on each Channel
template MulScalarCHW(C, H, W) {
    signal input in[C][H][W];
    signal input scalars[1][C][1][1];
    signal output out[C][H][W];

    for (var i=0; i < C; i++) {
        for (var j=0; j < H; j++) {
            for (var k=0; k < W; k++) {
                out[i][j][k] <== in[i][j][k] * scalars[0][i][0][0];
            }
        }
    }
}

template AddScalar(iShape, sc) {
    signal input in[iShape];
    signal output out[iShape];

    for (var i=0; i < iShape; i++) {
        out[i] <== in[i] + sc;
    }

}

template SubScalar(iShape, sc) {
    signal input in[iShape];
    signal output out[iShape];

    for (var i=0; i < iShape; i++) {
        out[i] <== in[i] - sc;
    }
}
