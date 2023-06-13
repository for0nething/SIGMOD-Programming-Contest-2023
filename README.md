# SIGMOD Programming Contest 2023

This is the code for champion solution of [ACM SIGMOD programming competition 2023](http://sigmod2023contest.eastus.cloudapp.azure.com/index.shtml).
The core idea of our approach can be found in [our poster](http://sigmod2023contest.eastus.cloudapp.azure.com/index.shtml).

Using this code, an almost fully accurate K-nearest-neighbor graph can be built for tens of millions of high-dimensional vectors in a very short time.

## Description
The code is based on the open-source implementation of NN-descent, [KGraph](https://github.com/aaalgo/kgraph), and has greatly improved both the effectiveness and efficiency of the original algorithm. 
This enhanced code was then leveraged to tackle the contest problem.

## Folder Structure

    .
    ├── nn-descent                          # An optimized implementation of nn-descent
    ├── knn-construction-kgraph.cc          # Code for solving the contest problem
    ├── io.h                                # Load dataset and save the result
    ├── run.sh                              # Shell to automatically compile and run the code
    ├── makefile                            # The makefile to compile the contest code
    └── README.md                           

## Quick Start
- **Step 1:** Install necessary dependencies: IntelMKL, Eigen, openMP, and Boost.
    - `Intel MKL`: The [2023.0 version](https://www.intel.com/content/www/us/en/docs/onemkl/get-started-guide/2023-0/overview.html) was used in the contest. Some related environment variables need to be properly set. 
        I list the ones I used here as a reference:
        - `export LD_LIBRARY_PATH=/home/jiayi/disk/mklBLAS/mkl/latest/lib/intel64:$LD_LIBRARY_PATH`
        - `export C_INCLUDE_PATH=/home/jiayi/disk/mklBLAS/mkl/latest/include:$C_INCLUDE_PATH`
        - `export MKLROOT=/home/jiayi/disk/mklBLAS/mkl/latest/`
        - `export MKL_LIBRARY_DIRS=/home/jiayi/disk/mklBLAS/mkl/latest/lib/intel64`
    - `Eigen`: [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) is a C++ template library for linear algebra. It uses a MPL2 license. Since it is a header-only library, you can easily install it following the steps on its website.
    - `OpenMP`: We used the OpenMP 4.5 in the contest. For Ubuntu, it should have been pre-installed.
    - `Boost`: [Boost](https://www.boost.org/) provides free peer-reviewed portable C++ source libraries. We used the Boost 1.65.1 in the contest.
    - **Note:** The parameters in `makefile` and `/nn-descent/CMakeLists.txt` that are related to the paths of MKL and Eigen  need to be set according to the actual situation.
- **Step 2:** Install the optimized nn-descent implementation.
    - `cd nn-descent`
    - `./autoInstall.sh`

- **Step 3:** Compile and run the code to tackle the contest problem. Please first switch to the root directory of this code.
    - `./run.sh`
    - Note that the compiled program accepts the same arguments as the example code provided in the contest. 
    So you can also change datasets in the same way.
    - The parameters that obtained the best result in the contest are already set in `knn-construction-kgraph.cc`. So you do not need to change anything.
