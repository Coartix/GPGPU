# -------------- Security Verifications --------------
#
# Check if filter_impl_curr.cpp is in cpp-version/
# Check if filter_impl_curr.cu is in cuda-version/
#
# Check if filter_impl_curr.cpp is similar to filter_impl.cpp
# Check if filter_impl_curr.cu is similar to filter_impl.cu
#
# Check if benchmark.log exist, if exists then remove, else create
#
# -------------- Execute CPP versions --------------
# Write -- cpp --
# Activate cpp (remove .so, cmake, link cpp to so)
# Loop through all files in cpp-version:
#       replace file with filter_impl.cpp
#       execute code, write to benchmark.log (format: filename=time)
#
# Reset filter_impl.cpp to filter_impl_curr.cpp
#
# Write newline to benchmark.log
#
# -------------- Execute CUDA versions --------------
#
# Write -- cu --
# Activate cu (remove .so, cmake, link cu to so)
# Loop through all files in cu-version:
#       replace file with filter_impl.cu
#       execute code, write to benchmark.log (format: filename=time)
#
# Reset filter_impl.cu to filter_impl_curr.cu

if [ -d "stats/" ]; then
    rm -rf stats/
fi

mkdir stats

echo "Version; Time; Program" >> stats/benchmark.csv

for folder in "cu-version"/*/; do
    filename=$(echo "$folder" | sed 's/cu-version\///')
    folder="${filename%/}"
    filter_impl="$folder/filter_impl.cu"
    kernel="$folder/kernel.cu"
    cuda_utils="$folder/cuda_utils.cu"
    sh utils/execute_version.sh "$filter_impl" "$kernel" "$cuda_utils"
done

for folder in "cpp-version"/*/; do
    filename=$(echo "$folder" | sed 's/cpp-version\///')
    folder="${filename%/}"
    filter_impl="$folder/filter_impl.cpp"
    sh utils/execute_version.sh "$filter_impl"
done
