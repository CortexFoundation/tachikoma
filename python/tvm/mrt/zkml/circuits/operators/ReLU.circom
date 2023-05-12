pragma circom 2.0.3;

include "../util.circom";
include "../circomlib/compconstant.circom";
include "../circomlib/switcher.circom";

// ReLU layer
template ReLU () {
    signal input in;
    signal output out;

    component isPositive = IsPositive();

    isPositive.in <== in;
    
    out <== in * isPositive.out;
}

template ReLU1D (c) {
    signal input in[c];
    signal output out[c];

    component isPositive[c];

    for (var i=0; i < c; i++) {
        isPositive[i] = IsPositive();
        isPositive[i].in <== in[i];
        out[i] <== in[i] * isPositive[i].out;
    }
}

template ReLU2D (c,h) {
    signal input in[c][h];
    signal output out[c][h];

    component isPositive[c][h];

    for (var i=0; i < c; i++) {
        for (var j=0; j < h; j++) {
            isPositive[i][j] = IsPositive();
            isPositive[i][j].in <== in[i][j];
	    out[i][j] <== in[i][j] * isPositive[i][j].out;
        }
    }
}

template ReLU3D (c,h,w) {
    signal input in[c][h][w];
    signal output out[c][h][w];

    component isPositive[c][h][w];

    for (var i=0; i < c; i++) {
        for (var j=0; j < h; j++) {
            for (var k=0; k < w; k++) {
                isPositive[i][j][k] = IsPositive();
                isPositive[i][j][k].in <== in[i][j][k];
	        out[i][j][k] <== in[i][j][k] * isPositive[i][j][k].out;
            }
        }
    }
}

template Clip(iShape, min, max) {
    signal input in[iShape];
    signal output out[iShape];

    component ltmin[iShape];
    component swmin[iShape];
    component ltmax[iShape];
    component swmax[iShape];
    for (var i=0; i < iShape; i++) {
        // if in[i] < min, min
        // if in[i] >= min, in[i]
        ltmin[i] = LessThan_Full();
        ltmin[i].a <== in[i];
        ltmin[i].b <== min;

        swmin[i] = Switcher();
        swmin[i].sel <== ltmin[i].out;
        swmin[i].L <== in[i];
        swmin[i].R <== min;

        // if result < max, result
        // if result >= max, max
        ltmax[i] = LessThan_Full();
        ltmax[i].a <== swmin[i].outL;
        ltmax[i].b <== max;

        swmax[i] = Switcher();
        swmax[i].sel <== ltmax[i].out;
        swmax[i].L <== max;
        swmax[i].R <== swmin[i].outL;

        out[i] <== swmax[i].outL;
    }
}

/*
template Clip(iShape, min, max) {
    signal input in[iShape];
    signal output out[iShape];

    for (var i=0; i < iShape; i++) {
        component compMin = CompConstant(min);
        compMin.in <== in[i];

        switcherMin = Switcher();
        switcher.sel <== compMin.out;
        switcher.L <== min;
        switcher.R <== in[i];

        component compMax = CompConstant(max);
        compMax.in <== switcher.outL;

        switcherMax = Switcher();
        switcherMax.sel = compMax.out;
        switcher.L <== in[i];
        switcher.R <== max;

        out[i] <== switcher.outL;
    }
}
*/
