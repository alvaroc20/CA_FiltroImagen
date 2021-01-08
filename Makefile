CFLAGS=-Wno-deprecated-gpu-targets -O2 -Xcompiler -fopenmp -std=c++11
LDLIBS=`pkg-config --cflags --libs opencv4`
HILOS:=32
nombreImagen:="newyork.jpeg"
compile:
	nvcc filtro.cu -o filtro $(CFLAGS) $(LDLIBS)

run:
	./filtro $(HILOS) $(nombreImagen)

cleanPhoto:
	find -name "*_sobel.png" -delete

cleanExe:
	find -name "filtro" -delete

clean:
	find -name "*_sobel.png" -delete
	find -name "filtro" -delete
