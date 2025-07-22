# Build object files.
g++ -std=c++17 -c -fPIC ref_daxpy.cpp -o ref_daxpy.o
g++ -std=c++17 -c -fPIC ref_dgemv.cpp -o ref_dgemv.o
g++ -std=c++17 -c -fPIC ref_dgemm.cpp -o ref_dgemm.o
g++ -std=c++17 -c -fPIC ref_axpyt.cpp -o ref_axpyt.o
g++ -std=c++17 -c -fPIC ref_gemvt.cpp -o ref_gemvt.o
g++ -std=c++17 -c -fPIC ref_gemmt.cpp -o ref_gemmt.o
g++ -std=c++17 -c -fPIC hw3_utils.cpp -o hw3_utils.o

# Build shared library.
g++ -shared -o librefBLAS.so ref_daxpy.o ref_dgemv.o ref_dgemm.o ref_axpyt.o ref_gemvt.o ref_gemmt.o hw3_utils.o