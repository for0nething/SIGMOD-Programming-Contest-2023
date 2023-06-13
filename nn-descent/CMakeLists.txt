cmake_minimum_required (VERSION 2.6)
project (KGRAPH)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}    -O3 -std=c++11 -fPIC   -march=native -ffast-math -flto ")

# New for Eigen
find_package(Eigen3 REQUIRED NO_MODULE)


find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Boost REQUIRED COMPONENTS timer chrono system program_options )

set(MKL_INTERFACE "lp64")
set(LIB_SRC kgraph.cpp)
# add the include directories
include_directories(
		${KGRAPH_SOURCE_DIR}
		/usr/local/include/eigen3
		${MKLROOT}/include
)

# add the compiler options
add_compile_options(
		-DIN_PARALLEL
		-fopenmp
		#-Wall
		-march=native
		-ffast-math
		-flto
		-DNDEBUG
		-fopt-info-vec-optimized

)

# find the Intel MKL libraries
find_package(MKL REQUIRED)


set(MKL_LIBRARIES mkl_intel_lp64 mkl_core mkl_gnu_thread pthread m dl)

# link the Intel MKL libraries
link_directories(${MKL_LIBRARY_DIRS})
link_libraries(
		${MKL_LIBRARIES}
		pthread
		m
		dl
)

add_library(kgraph SHARED ${LIB_SRC})
add_library(kgrapha STATIC ${LIB_SRC})
set_target_properties(kgrapha PROPERTIES OUTPUT_NAME kgraph)


set(TOOLS )
foreach(TOOL ${TOOLS})
	target_link_libraries(${TOOL} kgrapha ${Boost_LIBRARIES} rt)
endforeach(TOOL)
# New for Eigen
target_link_libraries(kgrapha  Eigen3::Eigen ${MKL_LIBRARIES} ${Boost_LIBRARIES} rt)

install (FILES kgraph.h DESTINATION include)
install (TARGETS kgraph kgrapha
	 LIBRARY DESTINATION lib
	 ARCHIVE DESTINATION lib)
