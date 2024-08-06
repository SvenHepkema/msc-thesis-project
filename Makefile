CLANG_FLAGS = -std=c++17

# For the fast compilations:
DATA_TYPE=int8_t
VALUE_BIT_WIDTH=3

fast: src/main.cpp 
	clang++ $< -O3 -o bin/$@ $(CLANG_FLAGS) -DDATA_TYPE=$(DATA_TYPE) -DVBW=$(VALUE_BIT_WIDTH)

executable: src/main.cpp 
	clang++ $< -O3 -o bin/$@ $(CLANG_FLAGS)

ub-sanitizer: src/main.cpp 
	clang++ $< -fsanitize=undefined -O3 -o bin/$@ $(CLANG_FLAGS) -g -DDATA_TYPE=$(DATA_TYPE) -DVBW=$(VALUE_BIT_WIDTH)
