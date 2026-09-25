NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64 -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc
EXE	        = solar
OBJ	        = solar.o

default: $(EXE)

solar.o: solar_temp.cu
	$(NVCC) -c -o $@ solar_temp.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
