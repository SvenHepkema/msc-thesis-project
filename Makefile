CLANG_FLAGS = -std=c++17 -g -Weverything -Wno-c++98-compat-local-type-template-args

# For the fast compilations:
DATA_TYPE=int8_t
VALUE_BIT_WIDTH=3

fast: src/main.cpp 
	clang++ $< -O3 -o bin/$@ $(CLANG_FLAGS) -DDATA_TYPE=$(DATA_TYPE) -DVBW=$(VALUE_BIT_WIDTH)

executable: src/main.cpp 
	clang++ $< -O3 -o bin/$@ $(CLANG_FLAGS)

ub-sanitizer: src/main.cpp 
	clang++ $< -fsanitize=undefined -O3 -fno-omit-frame-pointer -o bin/$@ $(CLANG_FLAGS) -g -DDATA_TYPE=$(DATA_TYPE) -DVBW=$(VALUE_BIT_WIDTH)

address-sanitizer: src/main.cpp 
	clang++ $< -fsanitize=address -O3 -fno-omit-frame-pointer -o bin/$@ $(CLANG_FLAGS) -g -DDATA_TYPE=$(DATA_TYPE) -DVBW=$(VALUE_BIT_WIDTH)
