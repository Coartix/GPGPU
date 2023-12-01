#!/bin/bash

# if [ "$#" -ne 2 ]; then
#     echo "Usage: $0 <cpp|cu> <filename>"
#     exit 1
# fi

if [ "$1" != "cpp" ] && [ "$1" != "cu" ]; then
    echo "Error: Extension must be 'cpp' or 'cu'"
    exit 1
fi

cd ..

if [ "$#" -eq 4 ]; then
    if [ -e "benchmarking/$1-version/$3" ]; then
        mv "src/kernel.cu" "src/kernel_ref.cu"
        cp "benchmarking/$1-version/$3" "src/kernel.cu"
    fi
    if [ -e "benchmarking/$1-version/$4" ]; then
        mv "src/cuda_utils.cu" "src/cuda_utils_ref.cu"
        cp "benchmarking/$1-version/$4" "src/cuda_utils.cu"
    fi
fi


mv "src/filter_impl.$1" "src/filter_impl_ref.$1"

cp "benchmarking/$1-version/$2" "src/filter_impl.$1"
