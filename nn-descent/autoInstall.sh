rm CMakeCache.txt
cmake -DCMAKE_BUILD_TYPE=release .
make clean
make
sudo make install
