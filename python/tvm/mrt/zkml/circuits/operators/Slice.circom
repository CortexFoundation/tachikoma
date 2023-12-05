pragma circom 2.1.0;

template StrideSlice2D (shp1, shp2, b1, b2, e1, e2, stride) {
    signal input in[shp1][shp2];
    signal output out[(e1-b1)\stride][(e2-b2)\stride];
    for (var i = b1; i < e1; i+=stride) {
        for (var j = b2; j < e2; j+=stride) {
	    out[(i-b1)\stride][(j-b2)\stride] <== in[i][j];
        }
    }
}
