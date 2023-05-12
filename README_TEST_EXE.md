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

## afterwards you can run with
`python test.py`

