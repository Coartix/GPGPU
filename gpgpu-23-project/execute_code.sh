#!/bin/bash

# Check if input argument
if [ $# -gt 0 ]; then
    input="$1"  # Use the provided value
else
    input="subject/camera.webm"  # Use the default value
fi

# Check if outut argument
if [ $# -gt 1 ]; then
    output="$2"  # Use the provided value
else
    output="output.mp4"  # Use the default value
fi

# nix-shell                                            # 1
cmake -S . -B build --preset release -D USE_CUDA=ON  # 2 (ou debug)
cmake --build build                                  # 2

# wget https://gstreamer.freedesktop.org/media/sintel_trailer-480p.webm # 3
export GST_PLUGIN_PATH=$(pwd)                                         # 4
ln -s ./build/libgstcudafilter-cu.so libgstcudafilter.so             # 5

# nvprof --csv --log-file benchmarking/log.csv
gst-launch-1.0 uridecodebin uri=file://"$(pwd)/$input" ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location="$output"
