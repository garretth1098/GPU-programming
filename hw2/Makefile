NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64 -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc
SRC	= iron.cu
EXE	        = iron
OBJ	        = iron.o

default: $(EXE)

$(OBJ): $(SRC)
	$(NVCC) -c -o $@ $(SRC) $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
