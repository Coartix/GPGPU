#!/usr/bin/make -f
# ===========================================

# Run this file with one of the following syntaxes:
# - `make [-j] [all]`: build all binaries
# - `make clean`: remove artifacts
# - `make test`: build and run all tests
# - `make test-BINARYNAME`: build and run a particular test

# Source files: each ".cu" file is an independent program
source_files := $(sort $(wildcard *.cu))

# Target binaries: we just remove the ".cu" from the filenames
binaries := $(patsubst %.cu,%,$(source_files))

# Tests: we generate "test-$BINNAME" targets from binary names
tests := $(patsubst %,test-%,$(binaries))


# Display the projected build process
$(info )
$(info Source files: $(source_files))
$(info Available binary targets: $(binaries))
$(info Available test targets: $(tests))
$(info )

# Rules
# -------------------------------------------
.PHONY: all
all: $(binaries)
	@echo "Target $@: COMPLETE"

.PHONY: all
clean:
	rm -f $(binaries)
	@echo "Target $@: COMPLETE"

# Each binary is based only on the source ".cu" file.
# We activate debugging information.
$(binaries): %: %.cu
	nvcc -o $@ -g -G $< 
	@echo "Target $@: COMPLETE"

.PHONY: test $(tests)
test: $(tests)
	@echo "Target $@: COMPLETE"
	@echo "Congratulations!"

# Each test is based only on the associated binary.
$(tests): test-%: %
	@echo ""
	@echo "Running test $@"
	@echo "----------------"
	./$<
	@echo "^^^^^^^^^^^^^^^^"
	@echo "Target $@: COMPLETE"
	@echo ""
