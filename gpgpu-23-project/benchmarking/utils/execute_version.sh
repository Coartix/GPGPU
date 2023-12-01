#!/bin/bash


echo "====================================================================="
echo "${1^^}"

# if [ "$#" -ne 1 ]; then
#     echo "Usage: $0 <filter_impl_code>"
#     exit 1
# fi

regex=".*/filter_impl\.(cpp|cu)"

# Check if the argument matches the regex
if ! [[ $1 =~ $regex ]]; then
    echo "<filter_impl_code> should match regex $regex"
    exit 1
else
    extension="${BASH_REMATCH[1]}"
fi

if [ "$#" -eq 3 ]; then
    sh utils/replace_file.sh "$extension" "$1" "$2" "$3"
else
    sh utils/replace_file.sh "$extension" "$1"
fi

# sh utils/replace_file.sh $extension $1

sh utils/set_env.sh "$extension"

cd ..
export GST_PLUGIN_PATH=$(pwd)

# start_time=$(date +%s.%N)
if [ -e "output.txt" ]; then
    rm output.txt
fi

echo "Executing $1..."
# nvprof --csv --log-file benchmarking/log.csv
#
transformed=$(echo "$1" | sed 's/\./_/g')
version="${transformed%/*}"

if [ "$extension" = "cu" ]; then
    echo "Profiling for $1..."
    nvprof --csv --log-file "benchmarking/stats/log_$version.csv" gst-launch-1.0 uridecodebin uri=file://$(pwd)/"subject/camera.webm" ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location="benchmarking/stats/video_cu_$version.mp4" >> output.txt
else
    gst-launch-1.0 uridecodebin uri=file://$(pwd)/"subject/camera.webm" ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location="benchmarking/stats/video_cpp_$version.mp4" >> output.txt
fi

execution_time=$(awk '/Execution ended after/ {print $NF}' output.txt)

rm output.txt
# end_time=$(date +%s.%N)
# execution_time=$(echo "$end_time - $start_time" | bc)

echo "Execution time: $execution_time seconds"

rm src/filter_impl.$extension
mv src/filter_impl_ref.$extension src/filter_impl.$extension

# Check if the file exists
if [ -e "src/cuda_utils_ref.cu" ]; then
    rm src/cuda_utils.cu
    mv src/cuda_utils_ref.cu src/cuda_utils.cu
fi

if [ -e "src/kernel_ref.cu" ]; then
    rm src/kernel.cu
    mv src/kernel_ref.cu src/kernel.cu
fi

cd -

echo "$version;$execution_time;$extension" >> stats/benchmark.csv
