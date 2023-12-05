pragma circom 2.1.0;

template Vision_GetValidCounts(i1, i2) {
    signal input in[i1][i2];
    signal input count;
    signal output out;
}
// calculate valid anchor counts
template TupleGetItem_VisCount_0(i1, i2) {
    signal input in[i1][i2];
    signal input threshold;
    signal output out1;
    for (var i = 0; i < i1; i++) {
      for (var j = 0; j < i2; j++) {
        if (in[i][1] >= threshold) {
	  out1 <== out1 + 1;
        }
      }
    }
}

// move anchor front
template TupleGetItem_VisCount_1(i1, i2) {
    signal input in[i1][i2];
    signal input threshold;
    signal output out[i1][i2];
    signal indice[i1];
    signal idx;
    idx <== 0;
    for (var i = 0; i < i1; i++) {
      if (in[i][1] >= threshold) {
        indice[i] <== idx;
        idx <== idx+1;
      }else{
        indice[i] <== i1 - i + idx - 1;
      }
      for (var j = 0; j < i2; j++) {
        out[indice[i]][j] <== in[i][j];
      }
    }
}

// calculate indice
template TupleGetItem_VisCount_2(i1, i2) {
    signal input in[i1][i2];
    signal input threshold;
    signal output indice[i1];
    signal idx;
    idx <== 0;
    for (var i = 0; i < i1; i++) {
      if (in[i][1] >= threshold) {
        indice[i] <== idx;
        idx <== idx+1;
      }else{
        indice[i] <== i1 - i + idx - 1;
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

    signal out_topK[topK][i2]; // valid to top first

    // 1. reserve topK score matrixs
    signal topK_indice[topK];
    for (var k = 0; k < topK; k++) {
      component mtmax[i1];
      component swmax[i1];
      component swmax_id[i1];

      signal now_max;
      now_max <== 0;
      signal now_max_id;
      now_max_id <== -1;
      for (var i = 0; i < in2; i++) {
        for (var g = 0; g < topK; g++) {
          if(i == topK_indice[g]){
            continue;
          }
        }
        mtmax[i] = MoreThan_Full();
        mtmax[i].a <== in1[i][1];
        mtmax[i].b <== now_max;

        swmax[i] = Switcher();
        swmax[i].sel <== mtmax[i].out;
        swmax[i].L <== in1[i][1];
        swmax[i].R <== now_max;
        now_max <== swmin[i].outL;

        swmax_id[i] = Switcher();
        swmax_id[i].sel <== mtmax[i].out;
        swmax_id[i].L <== i;
        swmax_id[i].R <== now_max_id;
        now_max_id <== swmin_id[i].outL;
      }

      // store result and move forward
      topK_indice[k] <== now_max_id;
      for (var j = 0; j < i2; j++) {
        out_topK[k][j] <== in1[now_max_id][j];
      }
    }

    // find same predict class matrix and overlap matrix as in one group, 
    // do nms(delete which overlap_with_best_matrix > threshold),
    // if deleted one matrix, loop again
    // finally, only reserve max_output_size matrixs in one group
    signal total_index;
    total_index <== 0;
    for (var x = 0; x < topK; x++) {
      signal temp_topK[topK][i2]; // reserve in one group
      signal group_size;
      group_size <== 0;
      largest_size <== 0;
      for (var k = 0; k < topK; k++) {
        if(group_size == 0){
          if(out_topK[k][1] != 0){
            for (var j = 0; j < i2; j++) {
              temp_topK[group_size][j] <== out_topK[k][j]
            }
            out_topK[k][1] <== 0;
            group_size <== group_size + 1;
          }
        }
        else {
          // put in if within overlap

        }
      }

      // put best matrix first
      signal first_best;
      first_best <== temp_topK[0][1];
      signal first_best_id;
      first_best_id <== 0;
      for (var k = 1; k < group_size; k++) {
        if(temp_topK[k][1]>first_best){
          first_best <== temp_topK[k][1];
          first_best_id <== k;
        }
      }
      signal matrix_temp[i2];
      for (var j = 0; j < i2; j++) {
        matrix_temp[j] <== temp_topK[0][j];
      }
      for (var j = 0; j < i2; j++) {
        temp_topK[0][j] <== temp_topK[first_best_id][j];
      }
      for (var j = 0; j < i2; j++) {
        temp_topK[first_best_id][j] <== matrix_temp[j];
      }

      // compare group overlap with max, and remove over threshold
      for (var k = 1; k < group_size; k++) {
        temp_topK[0][2] - temp_topK[k][2] - temp_topK[0][3] + temp_topK[k][3]
      }

      // move to final output
      signal count_out;
      count_out <== 0;
      for (var i = 0; i < group_size; i++) {
        if(temp_topK[i][1]>=0){
          for (var j = 0; j < i2; j++) {
            out[total_index][j] <== temp_topK[i][j];
          }
          total_index <== total_index+1;
        }
        count_out <== count_out+1;
        if (count_out >= max_output_size){
          break;
        }
      }
    }
    // finished
}
