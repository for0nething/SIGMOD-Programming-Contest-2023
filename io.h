
/**
 *  Example code for IO, read binary data vectors and write knng to path.
 *
 */

#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "assert.h"
#include <Eigen/Dense>
using std::cout;
using std::endl;
using std::string;
using std::vector;

/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knng a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
void SaveKNNG(const std::vector<std::vector<uint32_t>> &knng,
              const std::string &path = "output.bin") {
  std::ofstream ofs(path, std::ios::out | std::ios::binary);
  const int K = 100;
  const uint32_t N = knng.size();
  std::cout << "Saving KNN Graph (" << knng.size() << " X 100) to " << path
            << std::endl;
//  cout<<"knng.front().size()" << knng.front().size()<<"\n";
  assert(knng.front().size() == K);
  for (unsigned i = 0; i < knng.size(); ++i) {
    auto const &knn = knng[i];
    ofs.write(reinterpret_cast<char const *>(&knn[0]), K * sizeof(uint32_t));
  }
  ofs.close();
}

/// @brief Reading binary data vectors. Raw data store as a (N x 100)
/// binary file.
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadBin(const std::string &file_path,
             std::vector<std::vector<float>> &data) {
  std::cout << "Reading Data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  uint32_t N;  // num of points

  ifs.read((char *)&N, sizeof(uint32_t));
  data.resize(N);
  std::cout << "# of points: " << N << std::endl;

  const int num_dimensions = 100;
  std::vector<float> buff(num_dimensions);
  int counter = 0;
  while (ifs.read((char *)buff.data(), num_dimensions * sizeof(float))) {
    // new: pad 0
//    std::vector<float> row(num_dimensions + 4);

    std::vector<float> row(num_dimensions);
    for (int d = 0; d < num_dimensions; d++) {
      row[d] = static_cast<float>(buff[d]);
    }

    // new: pad 0
//    for (int i =0;i<4;i++)
//      row[100 + i] = 0;
    data[counter++] = std::move(row);
  }

  ifs.close();
  std::cout << "Finish Reading Data" << endl;
}

/// Read Bin Data for Eigen
/// @brief Reading binary data vectors. Raw data store as a (N x 100)
/// binary file.
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadBinEigen(const std::string &file_path,
             Eigen::MatrixXf&  data) {
  std::cout << "Reading Data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  uint32_t N;  // num of points

  ifs.read((char *)&N, sizeof(uint32_t));
  data = Eigen::MatrixXf((int)N, 100);
  std::cout << "# of points: " << N << std::endl;
  const int num_dimensions = 100;
  std::vector<float> buff(num_dimensions);
  int counter = 0;
  while (ifs.read((char *)buff.data(), num_dimensions * sizeof(float))) {
    for (int d = 0; d < num_dimensions; d++) {
      data(counter, d) = static_cast<float>(buff[d]);
    }
    ++counter;
  }

  ifs.close();
  std::cout << "Finish Reading Data" << endl;
}


void ReadBinEigenColMajor(const std::string &file_path,
                  Eigen::MatrixXf&  data) {
  std::cout << "Reading Data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  uint32_t N;  // num of points

  ifs.read((char *)&N, sizeof(uint32_t));
  data = Eigen::MatrixXf(100, (int)N);
  std::cout << "# of points: " << N << std::endl;
  const int num_dimensions = 100;
  std::vector<float> buff(num_dimensions);
  int counter = 0;
  while (ifs.read((char *)buff.data(), num_dimensions * sizeof(float))) {
    for (int d = 0; d < num_dimensions; d++) {
      data(d, counter) = static_cast<float>(buff[d]);
    }
    ++counter;
  }

  ifs.close();
  std::cout << "Finish Reading Data" << endl;
}

