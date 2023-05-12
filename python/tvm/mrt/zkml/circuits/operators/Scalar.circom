pragma circom 2.0.3;

template MulScalar(iShape, sc) {
    signal input in[iShape];
    signal output out[iShape];

    for (var i=0; i < iShape; i++) {
        out[i] <== in[i] * sc;
    }
}

template MulScalar_b(H) {
    signal input in[H];
    signal input sc;
    signal output out[H];

    for (var i=0; i < H; i++) {
        out[i] <== in[i] * sc;
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
