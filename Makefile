# For Gust
# module purge
# module load ncarenv/23.03 craype/2.7.20 nvhpc/23.1 cuda/11.7.1 cray-mpich/8.1.25 ncarcompilers/0.8.0
NVCC      = nvcc
MPICC     = mpicxx
MPIINCDIR = /opt/cray/pe/mpich/8.1.25/ofi/nvidia/20.7/include
NVCCFLAGS = -O2 -m64 -std=c++11 -g -I$(MPIINCDIR)
CFLAGS    = -O2 -std=c++11 -g
LDFLAGS   = -O2 -m64 -lcudart -g
OBJ       = main.o gpu_driver.o hello.o 


.SUFFIXES:
.SUFFIXES: .o .cpp .cu

hello: $(OBJ)
	$(MPICC) -o hello $(LDFLAGS) $(OBJ)

%.o: %.cpp
	$(MPICC) -c $(CFLAGS) $< 

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $<

clean:
	rm -f hello $(OBJ)
