#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_AVX512
#define EIGEN_DONT_PARALLELIZE


#include <sys/time.h>
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <vector>
#include "assert.h"
#include "io.h"
#include "nn-descent/kgraph.h"
#include <omp.h>
// for avx
#include <x86intrin.h>

// for timer
#include <boost/timer/timer.hpp>
//#include "include/efanna2e/index_kdtree.h"
//#define timer timer_for_boost_progress_t



// for Eigen
#include <Eigen/Dense>
using namespace boost;

#ifdef __AVX__
#define KGRAPH_MATRIX_ALIGN 32
#endif
using std::cout;
using std::endl;
using std::string;
using std::vector;
using namespace kgraph;

#define _INT_MAX 2147483640



// Modify for Eigen
typedef Eigen::MatrixXf MyType;

const int DIM = 100;
const int K = 100;


typedef float ResultType;



// 【Eigen dist】
float compare(const MyType& a, const MyType& b) {
//  return ((a - b)*(a - b).transpose())(0,0);
  return ((a - b).transpose() *(a-b))(0,0);
}

// 【Eigen dist id】
float compare_with_id(const MyType& a, const MyType& b, uint32_t id_a, uint32_t id_b) {
//  return (urn ((a - b)*(a - b).transpose())(0,0);
//  return ((a - b).transpose() *(a-b))(0,0);
//  Eigen::MatrixXf tmp = (- 2 * (a.transpose()) * b);
//  cout<<KGraph::square_sums.size() <<" "<<KGraph::square_sums.rows() <<" "<<KGraph::square_sums.cols()<<"\n";
//  float ret = KGraph::square_sums(id_a, 0);
//  float ret = KGraph::square_sums(id_a, 0) + KGraph::square_sums(id_b, 0) -2 * (a.transpose()) * b)(0,0);
//  return ret;
  return (KGraph::square_sums(id_a, Eigen::all) + KGraph::square_sums(id_b, Eigen::all) + (((-2 * a.transpose()) * b)))(0,0);
}



// 【Eigen version】
typedef kgraph::VectorOracle<MyType, MyType> MyOracle;



int main(int argc, char **argv) {
  boost::timer::cpu_timer timer;
  string source_path = "dummy-data.bin";
//  string source_path = "contest-data-release-1m.bin";
//  string source_path = "contest-data-release-10m.bin";


  // Also accept other path for source data
  if (argc > 1) {
    source_path = string(argv[1]);
  }
  omp_set_num_threads(32);

  // Read data points
//  ReadBinEigen(source_path, KGraph::nodes);   // Eigen version
  ReadBinEigenColMajor(source_path, KGraph::nodes);   // Eigen version


  cout<<KGraph::nodes.cols()<<"\n";
  int n =  KGraph::nodes.cols();  // note: this should be rows rather than size!


  // K-graph related
  MyOracle oracle(KGraph::nodes, compare, compare_with_id);


  KGraph *index = KGraph::create();



  KGraph::IndexParams params;

  params.S = 100;
  params.K = 100;
  params.L=  265;
  params.R = 350;
  params.iterations= 8;


  params.recall = 1.0;
  params.delta = 0.0002;

  // 【For submit】
//  params.if_eval = false;
//  params.controls = 0;

  // 【For local evaluation】
  params.if_eval = true;
  params.controls= 100;


  index->build(oracle, params);

  printf("Build finished!\n");
  auto times = timer.elapsed();
  std::cerr << "Build time: " << times.wall / 1e9 <<"\n";


  auto times_get_knng = timer.elapsed();
  std::cerr << "Get KNNG time: " << (times_get_knng.wall - times.wall) / 1e9 <<"\n";


  // Save to ouput.bin
  SaveKNNG(index->knng);
  auto times_save = timer.elapsed();
  std::cerr << "Save time: " << (times_save.wall - times_get_knng.wall) / 1e9 << "\n";

  return 0;
}
