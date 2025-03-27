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
CUDA_FLAGS = -ccbin /usr/bin/clang++-14 $(OPTIMIZATION_LEVEL) --resource-usage  -arch=sm_$(COMPUTE_CAPABILITY) -I $(CUDA_LIBRARY_PATH)/include -I. -L $(CUDA_LIBRARY_PATH)/lib64 -lcudart -lcurand -lcuda -lineinfo $(INC) $(LIB) --expt-relaxed-constexpr  -Xcompiler "$(CUDA_WARNING_FLAGS)" -diag-suppress $(NVCC_IGNORE_ERR_NUMBERS)


HEADER_FILES=$(wildcard src/**.h) $(wildcard src/cpu/*.cuh) $(wildcard src/gpu-kernels/*.hpp) $(wildcard src/common/*.hpp)
GPU_CUH := $(wildcard src/gpu-kernels/*.cuh)  $(wildcard src/common/*.hpp)

GPU_OBJ := $(patsubst src/gpu-kernels/%.cu, obj/gpu-kernels-%.o, $(wildcard src/gpu-kernels/*.cu))
FLS_OBJ := $(patsubst src/fls/%.cpp, obj/fls-%.o, $(wildcard src/fls/*.cpp))
ALP_OBJ := $(patsubst src/alp/%.cpp, obj/alp-%.o, $(wildcard src/alp/*.cpp))

# OBJ Files
obj/fls-%.o: src/fls/%.cpp
	clang++ $^  -c -o $@ $(CLANG_FLAGS)

obj/alp-%.o: src/alp/%.cpp
	clang++ $^  -c -o $@ $(CLANG_FLAGS)

obj/gpu-kernels-%.o: src/gpu-kernels/%.cu  $(GPU_CUH)
	nvcc $(CUDA_FLAGS) -c -o $@ $(word 1, $^)

# Executables
SOURCE_FILES=src/main.cpp $(FLS_OBJ) $(ALP_OBJ) $(GPU_OBJ)

executable: $(SOURCE_FILES) $(HEADER_FILES)
	clang++ $(SOURCE_FILES) $(OPTIMIZATION_FLAG) -o bin/$@ $(CLANG_FLAGS) $(CUDA_OBJ_FLAGS) 

ub-sanitizer: $(SOURCE_FILES) $(HEADER_FILES)
	clang++ $(SOURCE_FILES) -O2 -fsanitize=undefined  -fno-omit-frame-pointer -o bin/$@ $(CLANG_FLAGS) $(CUDA_OBJ_FLAGS)  -g 

address-sanitizer: $(SOURCE_FILES) $(HEADER_FILES)
	clang++ $(SOURCE_FILES) -O3 -fsanitize=address  -fno-omit-frame-pointer -o bin/$@ $(CLANG_FLAGS) $(CUDA_OBJ_FLAGS) -g 

clean:
	rm -f bin/*
	rm -f obj/*
