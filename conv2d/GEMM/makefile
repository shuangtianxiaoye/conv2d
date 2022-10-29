src = $(wildcard ./*.cpp)

obj = $(patsubst %.cpp, %.o, $(src))

myArgs = -Wall -msse -fopenmp -O2

all:GEMM

GEMM:$(obj)
	g++ $^ -o $@ $(myArgs)

$(obj):%.o:%.cpp
	g++ -c $< -o $@ $(myArgs)

run:
	make clean
	make all
	./GEMM >> output.txt

clean:
	-rm -rf GEMM output.txt $(obj)

.PHONY:ALL run clean