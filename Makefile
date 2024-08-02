CLANG_FLAGS = -std=c++17

first: src/first.cpp 
	clang++ $< -O3 -o bin/$@ $(CLANG_FLAGS)
