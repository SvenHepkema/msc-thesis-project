INC := -I $(CUDA_LIBRARY_PATH)/include -I.
LIB := -L $(CUDA_LIBRARY_PATH)/lib64 -lcudart -lcurand 
CUDA_OBJ_FLAGS = $(INC) $(LIB) 
OPTIMIZATION_LEVEL = -O3
CLANG_FLAGS = -std=c++17 -g $(WARNINGS) $(OPTIMIZATION_LEVEL)
WARNINGS = -Weverything -Wno-c++98-compat-local-type-template-args -Wno-c++98-compat-pedantic -Wno-c++98-compat -Wno-padded -Wno-float-equal -Wno-global-constructors -Wno-exit-time-destructors 

#3033-D: inline variables are a C++17 feature
#3356-D: structured bindings are a C++17 feature
NVCC_IGNORE_ERR_NUMBERS=3033,3356
CUDA_WARNING_FLAGS=-Wno-c++17-extensions
COMPUTE_CAPABILITY = 61
#CUSTOM_FLAGS = -DSingleVectorMapping
CUDA_FLAGS = --std c++17 -ccbin /usr/bin/clang++-14 $(OPTIMIZATION_LEVEL) --resource-usage  -arch=sm_$(COMPUTE_CAPABILITY) -I $(CUDA_LIBRARY_PATH)/include -I. -L $(CUDA_LIBRARY_PATH)/lib64 -lcudart -lcurand -lcuda -lineinfo $(INC) $(LIB) --expt-relaxed-constexpr  -Xcompiler "$(CUDA_WARNING_FLAGS)" -diag-suppress $(NVCC_IGNORE_ERR_NUMBERS) $(CUSTOM_FLAGS)

ENGINE_HEADER_FILES=$(wildcard src/engine/*.cuh)
FLSGPU_HEADER_FILES=$(wildcard src/flsgpu/*.cuh)
HEADER_FILES=src/alp/alp-bindings.cuh $(wildcard src/flsgpu/*.cuh) $(wildcard src/engine/*.cuh)

FLS_OBJ := $(patsubst src/fls/%.cpp, obj/fls-%.o, $(wildcard src/fls/*.cpp))
ALP_OBJ := $(patsubst src/alp/%.cpp, obj/alp-%.o, $(wildcard src/alp/*.cpp))
GENERATED_BINDINGS_OBJ := $(patsubst src/generated-bindings/%.cu, obj/generated-bindings-%.o, $(wildcard src/generated-bindings/*.cu))

# OBJ Files
obj/fls-%.o: src/fls/%.cpp
	clang++ $^  -c -o $@ $(CLANG_FLAGS)

obj/alp-%.o: src/alp/%.cpp
	clang++ $^  -c -o $@ $(CLANG_FLAGS)

obj/generated-bindings-%.o: src/generated-bindings/%.cu $(FLSGPU_HEADER_FILES) src/engine/device-utils.cuh src/engine/kernels.cuh
	nvcc $(word 1, $^) -c -o $@ $(CUDA_FLAGS) 

obj/enums.o: src/engine/enums.cu src/engine/enums.cuh
	nvcc $(word 1, $^) -c -o $@ $(CUDA_FLAGS) 

obj/alp-bindings.o: src/alp/alp-bindings.cu $(ALP_OBJ) 
	nvcc $(CUDA_FLAGS) -c -o $@ src/alp/alp-bindings.cu 

# Executables
SOURCE_FILES=obj/alp-bindings.o obj/enums.o $(FLS_OBJ) $(ALP_OBJ) $(GENERATED_BINDINGS_OBJ)

test: src/test.cu $(SOURCE_FILES) $(HEADER_FILES)
	nvcc $(CUDA_FLAGS) -g -o bin/$@ src/test.cu $(SOURCE_FILES)

clean:
	rm -f bin/*
	rm -f obj/*
