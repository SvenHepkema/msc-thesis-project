CLANG_FLAGS = -std=c++17

executable: src/main.cpp 
	clang++ $< -O3 -o bin/$@ $(CLANG_FLAGS)
