pragma circom 2.1.0;

include "../circomlib/switcher.circom";
include "../util.circom";

// just Pass, no need to output
template Vision_GetValidCounts(i1, i2) {
    signal input in[i1][i2];
    signal input count;
    signal output out;
}

// calculate valid anchor counts
template TupleGetItem_VisCount_0(i1, i2) {
    signal input in[i1][i2];
    signal input threshold;
    signal output out; // count of valid anchor

    component gethan[i1];
    component switcher[i1];

    for (var i = 0; i < i1; i++) {
        gethan[i] = GreaterEqThan_Full();
        gethan[i].a <== in[i][1];
        gethan[i].b <== threshold;

        switcher[i] = Switcher();
        switcher[i].sel <== gethan[i].out;
        switcher[i].L <== out + 1;
        switcher[i].R <== out;
        out <== switcher[i].outL;
    }
}

// move anchor front
template TupleGetItem_VisCount_1(i1, i2) {
    signal input in[i1][i2];
    signal input threshold;
    signal output out[i1][i2];
    signal indice[i1];

    signal idx[i1+1];
    idx[0] <== 0;
    component gethan[i1];
    component sw0[i1];
    component sw1[i1];

    // set the valid and invalid index
    for (var i = 0; i < i1; i++) {
        gethan[i] = GreaterEqThan_Full();
        gethan[i].a <== in[i][1];
        gethan[i].b <== threshold;

        sw0[i] = Switcher();
        sw0[i].sel <== gethan[i].out;
        sw0[i].L <== idx[i];
        sw0[i].R <== i1 - i + idx[i] - 1;
        indice[i] <== sw0[i].outL;

        sw1[i] = Switcher();
        sw1[i].sel <== gethan[i].out;
        sw1[i].L <== idx[i]+1;
        sw1[i].R <== idx[i];
        idx[i+1] <== sw1[i].outL;
    }

    // get the new output matrix
    component isequal[i1][i1];
    component switcher[i1][i1][i2];
    var tempout[i1][i1+1][i2];
    for (var i = 0; i < i1; i++) {
      for (var j = 0; j < i2; j++) {
        tempout[i][0][j] === 0; // initialize first matrix
      }
    }

    for (var i = 0; i < i1; i++) {
      for (var x = 0; x < i1; x++) {
        isequal[i][x] = IsEqual();
        isequal[i][x].in[0] <== x;
        isequal[i][x].in[1] <== indice[i];

        for (var j = 0; j < i2; j++) {
          switcher[i][x][j] = Switcher();
          switcher[i][x][j].sel <== isequal[i][x].out;
          switcher[i][x][j].L <== in[i][j];
          switcher[i][x][j].R <== tempout[i][x][j];
          tempout[i][x+1][j] === switcher[i][x][j].outL;
        }
      }

      for (var j = 0; j < i2; j++) {
        out[i][j] <== tempout[i][i1][j];
      }
    }
}

// calculate indice
template TupleGetItem_VisCount_2(i1, i2) {
    signal input in[i1][i2];
    signal input threshold;
    signal output indice[i1];

    /* for (var i = 0; i < i1; i++) { //pseudo-code
      if (in[i][1] >= threshold) {
        indice[i] <== idx;
        idx <== idx+1;
      }else{
        indice[i] <== i1 - i + idx - 1;
      }
    }*/

    signal idx[i1+1];
    idx[0] <== 0;
    component gethan[i1];
    component sw0[i1];
    component sw1[i1];

    // set the valid and invalid index
    for (var i = 0; i < i1; i++) {
        gethan[i] = GreaterEqThan_Full();
        gethan[i].a <== in[i][1];
        gethan[i].b <== threshold;

        sw0[i] = Switcher();
        sw0[i].sel <== gethan[i].out;
        sw0[i].L <== idx[i];
        sw0[i].R <== i1 - i + idx[i] - 1;
        indice[i] <== sw0[i].outL;

        sw1[i] = Switcher();
        sw1[i].sel <== gethan[i].out;
        sw1[i].L <== idx[i]+1;
        sw1[i].R <== idx[i];
        idx[i+1] <== sw1[i].outL;
    }
}

// move valid nms output front
template VisionNonMaxSuppression_Move_Valid_Forward_funcutil(i1, i2) {
    signal input in[i1][i2+1];
    signal output out[i1][i2];
    signal indice[i1];

    signal idx[i1+1];
    idx[0] <== 0;
    component gethan[i1];
    component sw0[i1];
    component sw1[i1];

    // set the valid and invalid index
    for (var i = 0; i < i1; i++) {
        gethan[i] = GreaterEqThan_Full();
        gethan[i].a <== in[i][i2];
        gethan[i].b <== 1;

        sw0[i] = Switcher();
        sw0[i].sel <== gethan[i].out;
        sw0[i].L <== idx[i];
        sw0[i].R <== i1 - i + idx[i] - 1;
        indice[i] <== sw0[i].outL;

        sw1[i] = Switcher();
        sw1[i].sel <== gethan[i].out;
        sw1[i].L <== idx[i]+1;
        sw1[i].R <== idx[i];
        idx[i+1] <== sw1[i].outL;
    }

    // get the new output matrix
    component isequal[i1][i1];
    component switcher[i1][i1][i2];
    var tempout[i1][i1+1][i2];
    for (var i = 0; i < i1; i++) {
      for (var j = 0; j < i2; j++) {
        tempout[i][0][j] === 0; // initialize first matrix
      }
    }

    for (var i = 0; i < i1; i++) {
      for (var x = 0; x < i1; x++) {
        isequal[i][x] = IsEqual();
        isequal[i][x].in[0] <== x;
        isequal[i][x].in[1] <== indice[i];

        for (var j = 0; j < i2; j++) {
          switcher[i][x][j] = Switcher();
          switcher[i][x][j].sel <== isequal[i][x].out;
          switcher[i][x][j].L <== in[i][j];
          switcher[i][x][j].R <== tempout[i][x][j];
          tempout[i][x+1][j] === switcher[i][x][j].outL;
        }
      }

      for (var j = 0; j < i2; j++) {
        out[i][j] <== tempout[i][i1][j];
      }
    }
}

// score_index = 1, sort front if score greater
template VisionNonMaxSuppression_Sort_Detection_SSD_funcutil(i1, i2) {
    signal input in[i1][i2]; // data
    signal output out[i1][i2];

    var temp_out[i1+1][i1][i2];
    var temp_out_valid[i1+1][i1][1];
    for (var i = 0; i < i1; i++) {
      for (var j = 0; j < i2; j++) {
        temp_out[0][i][j] === in[i][j];
      }
      temp_out_valid[0][i][0] === 0; // uncomment if i2+1
    }

    component visionNonMaxSuppression_move_valid_forward_funcutil_0[i1];
    var largest_indice[i1][i1+1];
    var largest_score[i1][i1+1];
    component gthan[i1][i1];
    component iseq_0[i1][i1];
    component sw_ind[i1][i1];
    component sw_sco[i1][i1];
    component sw_val[i1][i1];

    for (var x = 0; x < i1; x++) {
      // 1.1 find max
      largest_indice[x][0] === x;
      largest_score[x][0] === temp_out[x][x][1];
      for (var i = 1; i < i1-x; i++) {
        gthan[x][i] = GreaterThan_Full();
        gthan[x][i].a <== temp_out[x][i+x][1];
        gthan[x][i].b <== largest_score[x][i-1];

        sw_ind[x][i] = Switcher();
        sw_ind[x][i].sel <== gthan[x][i].out;
        sw_ind[x][i].L <== i+x;
        sw_ind[x][i].R <== largest_indice[x][i-1];
        sw_sco[x][i] = Switcher();
        sw_sco[x][i].sel <== gthan[x][i].out;
        sw_sco[x][i].L <== temp_out[x][i+x][1];
        sw_sco[x][i].R <== largest_score[x][i-1];

        largest_indice[x][i] === sw_ind[x][i].outL;
        largest_score[x][i] === sw_sco[x][i].outL;
      }

      // 1.2 set valid
      for (var i = 0; i < i1-x; i++) {
        iseq_0[x][i] = IsEqual();
        iseq_0[x][i].in[0] <== largest_indice[x][i1-x-1];
        iseq_0[x][i].in[1] <== i+x;

        sw_val[x][i] = Switcher();
        sw_val[x][i].sel <== iseq_0[x][i].out;
        sw_val[x][i].L <== 1;
        sw_val[x][i].R <== 0;

        temp_out_valid[x][i+x][0] === sw_val[x][i].outL;
      }

      // 2.1 copy ahead
      for (var i = 0; i < x; i++) {
        for (var j = 0; j < i2+1; j++) {
          temp_out[x+1][i][j] === temp_out[x][i][j];
        }
      }
      // 2.2 put max first, copy after
      visionNonMaxSuppression_move_valid_forward_funcutil_0[x] = VisionNonMaxSuppression_Move_Valid_Forward_funcutil(i1-x, i2);
      for (var i = 0; i < i1-x; i++) {
        for (var j = 0; j < i2; j++) {
          visionNonMaxSuppression_move_valid_forward_funcutil_0[x].in[i][j] <== temp_out[x][i+x][j];
        }
        visionNonMaxSuppression_move_valid_forward_funcutil_0[x].in[i][i2] <== temp_out_valid[x][i+x][0];
      }
      for (var i = 0; i < i1-x; i++) {
        for (var j = 0; j < i2; j++) {
          temp_out[x+1][i+x][j] === visionNonMaxSuppression_move_valid_forward_funcutil_0[x].out[i][j];
        }
      }
      temp_out[x+1][x][i2] === 1;
      for (var i = 1; i < i1-x; i++) {
        temp_out[x+1][i+x][i2] === 0;
      }

    }

    // copy result
    for (var i = 0; i < i1; i++) {
      for (var j = 0; j < i2; j++) {
        out[i][j] <== temp_out[i1][i][j];
      }
    }
}

template Vision_NonMaxSuppression(i1, i2, topK) {
    signal input in1[i1][i2]; // data
    signal input in2; // count
    signal input in3[i1]; // indice
    signal input in4; // max_output_size=-1
    signal input in5; // iou_threshold=0.5
    signal output out[i1][i2]; // valid to top

    signal out_topK[topK][i2]; // valid topK to top first
    // 1. reserve topK score matrixs, just sort
    component sort_out_0 = VisionNonMaxSuppression_Sort_Detection_SSD_funcutil(i1, i2);
    for (var i = 0; i < i1; i++) {
      for (var j = 0; j < i2; j++) {
        sort_out_0.in[i][j] <== in1[i][j];
      }
    }
    for (var i = 0; i < topK; i++) {
      for (var j = 0; j < i2; j++) {
        out_topK[i][i] <== sort_out_0.out[i][j];
      }
    }

    // 2. find same predict class matrix and overlap matrix as in one group,
    // make sure each matrix be added into group only once

    // 3. do nms(delete which overlap_with_best_matrix > threshold),
    // if deleted one matrix, loop goes on
    // finally, only reserve max_output_size matrixs in one group
    var nms_out[topK][i2+1]; // last column is valid or invalid
    //var nms_tempout[topK][topK][topK][i2+1]; // [each matrix][each matrix in a group may be deleted once][each matrix in a group][i2]

    // 4. move all valid to final output
    component visionNonMaxSuppression_move_valid_forward_funcutil_0 = VisionNonMaxSuppression_Move_Valid_Forward_funcutil(topK, i2);
    for (var i = 0; i < topK; i++) {
      for (var j = 0; j < i2+1; j++) {
        visionNonMaxSuppression_move_valid_forward_funcutil_0.in[i][j] <== nms_out[i][j];
      }
    }
    for (var i = 0; i < topK; i++) {
      for (var j = 0; j < i2; j++) {
        out[i][j] <== visionNonMaxSuppression_move_valid_forward_funcutil_0.out[i][j];
      }
    }

    // finished
}
