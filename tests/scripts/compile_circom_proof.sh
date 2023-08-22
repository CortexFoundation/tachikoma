#!/bin/bash
set -ex

circom circom_model_test.circom --r1cs --wasm --sym --c
cd circom_model_test_js
cp ../circom_model_test.json input.json
cp ../circom_model_test.r1cs .
cp ../circom_model_test.sym .

node generate_witness.js circom_model_test.wasm input.json witness.wtns

snarkjs powersoftau new bn128 18 pot12_0000.ptau -v  ##2**18 according to circom circuits scale
snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau --name="First contribution" -v ## enter text
snarkjs powersoftau prepare phase2 pot12_0001.ptau pot12_final.ptau -v  ##maybe most time-comsuming
snarkjs groth16 setup circom_model_test.r1cs pot12_final.ptau circom_model_test_0000.zkey
snarkjs zkey contribute circom_model_test_0000.zkey circom_model_test_0001.zkey --name="1st Contributor Name" -v
snarkjs zkey export verificationkey circom_model_test_0001.zkey verification_key.json
snarkjs groth16 prove circom_model_test_0001.zkey witness.wtns proof.json public.json
snarkjs groth16 verify verification_key.json public.json proof.json
snarkjs zkey export solidityverifier circom_model_test_0001.zkey verifier.sol
snarkjs generatecall
