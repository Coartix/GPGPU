#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <cpp|cu>"
    exit 1
fi

if [ "$1" != "cpp" ] && [ "$1" != "cu" ]; then
    echo "Usage: $0 <cpp|cu>"
    exit 1
fi

cd ..

rm libgstcudafilter.so

echo "Making Environment..."

cmake -S . -B build --preset release -D USE_CUDA=ON >> tmp.txt
cmake --build build > tmp.txt 2>&1

rm tmp.txt

echo "Linking Files..."
ln -s "./build/libgstcudafilter-$1.so" libgstcudafilter.so
