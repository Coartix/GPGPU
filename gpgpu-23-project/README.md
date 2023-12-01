### Set nix environment

nix-shell

### To Launch Code in src
execute:

#### SETUP CUDA
In execute\_code.sh change link line to
ln -s ./build/libgstcudafilter-cu.so libgstcudafilter.so

#### SETUP CPP
In execute\_code.sh change link line to
ln -s ./build/libgstcudafilter-cpp.so libgstcudafilter.so


##### Launch Code
`sh execute_code.sh`

### To launch Benchmark
#### Go to benchmarking folder
`cd benchmarking`

#### Launch Benchmark
`sh launch_benchmark.sh`

#### WARNING
##### To be able to to launch simple code in src/ after benchmarking,
##### remove build file and libgstcudafilter.so in folder irgpu/gpgpu-23-fall
`rm -rf build`
`rm libgstcudafilter.so`
