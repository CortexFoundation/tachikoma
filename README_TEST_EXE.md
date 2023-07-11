# tachikoma run test.py

## some libs or docs to refer to
[circom lib](https://github.com/iden3/circomlib)
[circom lib ml](https://github.com/socathie/circomlib-ml)
[circom origin](https://docs.circom.io/getting-started/installation/)
[circom zkp](https://docs.circom.io/background/background/)
[uchikoma](https://github.com/zk-ml/uchikoma)
[tachikoma](https://github.com/zk-ml/tachikoma)
[tvm nn](https://tvm.apache.org/docs/reference/api/python/relay/nn.html)
[llvm install](https://getting-started-with-llvm-core-libraries-zh-cn.readthedocs.io/zh_CN/latest/ch01.html#id6)
[llvm download](https://releases.llvm.org/download.html)
[cvm mrt ops](https://cvm-runtime.readthedocs.io/en/latest/deep_dive/math_formalization.html#cvm-cilp)

## first clone repo, must recursive !
`git clone --recursive https://github.com/zk-ml/tachikoma`

## then checkout cuda version
`nvidia-smi`

## install python env using conda
```bash
conda create --name tachikoma python=3.11 -c conda-forge
conda activate tachikoma
python -m pip install attrs pytest psutil scipy decorator typing_extensions
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
python -m pip install torchvision
python -m pip install torchaudio
```

## set up cmake config, using cuda&llvm-10
`cd tachikoma && git checkout wlt && vi cmake/config.cmake`

```cmake
set(USE_CUDA ON)
set(USE_LLVM ON)
```

## make tvm share libraries
`make -j24`

## must add local directory in test.py
```python
ROOT = path.dirname(__file__)
os.sys.path.insert(0, path.join(ROOT, "python"))
```

## adapt tvm.mrt to circom
......

## firstly it will download models, proxy is necessary 
`proxychains4 python test.py`

## afterwards you can run with, got code.circom and input.json
`python test.py`
`python scripts/image_scale_to_circom_input.py scripts/test_a.png input.json # resolve image input, and put in xxx.json`


# circom code usage
## compile and generate witness
first install 'nlohmann-json3-dev, libgmp-dev and nasm' in system
```bash
circom circom_model_test.circom --r1cs --wasm --sym --c
cd circom_model_test_cpp
make -j6
cp ../circom_model_test.json input.json
./circom_model_test input.json witness.wtns
# or in js
node generate_witness.js model.wasm input.json witness.wtns
```

## generate proof
```bash
npm install -g snarkjs
snarkjs powersoftau new bn128 18 pot12_0000.ptau -v  ##2**18 according to circom circuits scale
snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau --name="First contribution" -v  ## enter text
snarkjs powersoftau prepare phase2 pot12_0001.ptau pot12_final.ptau -v  ##maybe most time-comsuming
snarkjs groth16 setup circom_model_test.r1cs pot12_final.ptau circom_model_test_0000.zkey
ls
snarkjs zkey contribute circom_model_test_0000.zkey circom_model_test_0001.zkey --name="1st Contributor Name" -v  ## enter text
snarkjs zkey export verificationkey circom_model_test_0001.zkey verification_key.json
snarkjs groth16 prove circom_model_test_0001.zkey witness.wtns proof.json public.json
snarkjs groth16 verify verification_key.json public.json proof.json
snarkjs zkey export solidityverifier circom_model_test_0001.zkey verifier.sol
snarkjs generatecall
```

## using mypy to check tvm.mrt
```bash
python -m mypy -p python.tvm.mrt
```
