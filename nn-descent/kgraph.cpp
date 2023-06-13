#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_AVX512
//#define EIGEN_USE_THREADS 1
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_RUNTIME_NO_MALLOC


//#define SPECIFIC_TIME
//#define DIST_CNT
//#define SHOW_MEM_SIZE


#ifndef KGRAPH_VERSION
#define KGRAPH_VERSION unknown
#endif
#ifndef KGRAPH_BUILD_NUMBER
#define KGRAPH_BUILD_NUMBER 
#endif
#ifndef KGRAPH_BUILD_ID
#define KGRAPH_BUILD_ID
#endif
#define STRINGIFY(x) STRINGIFY_HELPER(x)
#define STRINGIFY_HELPER(x) #x

#ifdef _OPENMP
#include <omp.h>
#endif
#include <unordered_set>
#include <mutex>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <boost/timer/timer.hpp>
#define timer timer_for_boost_progress_t
#include <boost/progress.hpp>
#undef timer
#include <boost/dynamic_bitset.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include "boost/smart_ptr/detail/spinlock.hpp"
#include "kgraph.h"
#include <chrono>
#include <unordered_map>
//#include "kgraph-util.h"


#include <Eigen/Dense>
#include <bitset>

namespace kgraph {
    using namespace std;
    using namespace boost;
    using namespace boost::accumulators;

    struct Neighbor {
        uint32_t id;
        float dist;

        Neighbor() {}

        Neighbor(uint32_t i, float d) : id(i), dist(d) {
        }
    };
    typedef vector<Neighbor> Neighbors;
    static inline bool operator < (Neighbor const &n1, Neighbor const &n2) {
      return n1.dist < n2.dist;
    }
    static inline bool operator == (Neighbor const &n1, Neighbor const &n2) {
      return n1.id == n2.id;
    }


    typedef vector<Neighbor> Neighbors;
    Eigen::MatrixXf KGraph::nodes;
    Eigen::VectorXf KGraph::square_sums;




    uint32_t verbosity = default_verbosity;

    typedef boost::detail::spinlock Lock;
    typedef std::lock_guard<Lock> LockGuard;



    struct Control {
        uint32_t id;
        Neighbors neighbors;
    };


    // generate size distinct random numbers < N
    template<typename RNG>
    static void GenRandom(RNG &rng, uint32_t *addr, uint32_t size, uint32_t N) {
      if (N == size) {
        for (uint32_t i = 0; i < size; ++i) {
          addr[i] = i;
        }
        return;
      }
      for (uint32_t i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
      }
      sort(addr, addr + size);
      for (uint32_t i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
          addr[i] = addr[i - 1] + 1;
        }
      }
      uint32_t off = rng() % N;
      for (uint32_t i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
      }
    }

    static float EvaluateRecall(Neighbors const &pool, Neighbors const &knn) {
      unordered_map<uint32_t, bool> mp;
      for(const auto& z:knn)
        mp[z.id] = 1;
      uint32_t found = 0;
      for(const auto& z:pool){
        if(mp[z.id >> 1])++found;
      }
      return 1.0 * found / knn.size();
    }

    static float EvaluateAccuracy(Neighbors const &pool, Neighbors const &knn) {
      uint32_t m = std::min(pool.size(), knn.size());
      float sum = 0;
      uint32_t cnt = 0;
      for (uint32_t i = 0; i < m; ++i) {
        if (knn[i].dist > 0) {
          sum += abs(pool[i].dist - knn[i].dist) / knn[i].dist;
          ++cnt;
        }
      }
      return cnt > 0 ? sum / cnt : 0;
    }

    static float EvaluateOneRecall(Neighbors const &pool, Neighbors const &knn) {
      if (pool[0].dist == knn[0].dist) return 1.0;
      return 0;
    }


    // This function is now wrong
    static float EvaluateDelta(Neighbors const &pool, uint32_t K) {
      uint32_t c = 0;
      uint32_t N = K;
      if (pool.size() < N) N = pool.size();
      for (uint32_t i = 0; i < N; ++i) {
        ++c;
      }
      return float(c) / K;
    }


    // try insert nn into the list
    // the array addr must contain at least K+1 entries:
    //      addr[0..K-1] is a sorted list
    //      addr[K] is as output parameter
    // * if nn is already in addr[0..K-1], return K+1
    // * Otherwise, do the equivalent of the following
    //      put nn into addr[K]
    //      make addr[0..K] sorted
    //      return the offset of nn's index in addr (could be K)
    //
    // Special case:  K == 0
    //      addr[0] <- nn
    //      return 0

    // Insert a new point into the candidate pool in ascending order
    // Come from faiss
    template <typename NeighborT>
    uint32_t UpdateKnnListHelper(Neighbor* addr, uint32_t size, NeighborT nn) {
      // find the location to insert
      int left = 0, right = size - 1;
      if (addr[left].dist > nn.dist) {
        memmove((char*)&addr[left + 1], &addr[left], size * sizeof(Neighbor));
        addr[left] = nn;
        return left;
      }
      if (addr[right].dist < nn.dist) {
        addr[size] = nn;
        return size;
      }
      while (left < right - 1) {
        int mid = (left + right) / 2;
        if (addr[mid].dist > nn.dist)
          right = mid;
        else
          left = mid;
      }
      // check equal ID

      while (left > 0) {
        if (addr[left].dist < nn.dist)
          break;
        if ( (addr[left].id>>1) == nn.id)
          return size + 1;
        left--;
      }
      if ( (addr[left].id>>1) == nn.id || (addr[right].id>>1) == nn.id)
        return size + 1;
      memmove((char*)&addr[right + 1],
              &addr[right],
              (size - right) * sizeof(Neighbor));
      addr[right] = nn;
//      cout<<addr[right].id<<"  " << ((addr[right].id<<1)|1) <<"\n";
//      addr[right].id = (addr[right].id<<1)|1;
//      cout<<"after "<<addr[right].id<<"\n";
      return right;
    }

    static inline uint32_t UpdateKnnList (Neighbor *addr, uint32_t K, Neighbor nn) {
        return UpdateKnnListHelper<Neighbor>(addr, K, nn);
    }


    void LinearSearch (IndexOracle const &oracle, uint32_t i, uint32_t K, vector<Neighbor> *pnns) {
        vector<Neighbor> nns(K+1);
        uint32_t N = oracle.size();
        Neighbor nn;
        nn.id = 0;
//        nn.flag = true; // we don't really use this
        uint32_t k = 0;
        while (nn.id < N) {
            if (nn.id != i) {
                nn.dist = oracle(i, nn.id);
                UpdateKnnList(&nns[0], k, nn);
                if (k < K) ++k;
            }
            ++nn.id;
        }
        nns.resize(K);
        pnns->swap(nns);
    }


    void GenerateControl (IndexOracle const &oracle, uint32_t C, uint32_t K, vector<Control> *pcontrols) {
        vector<Control> controls(C);
        {
            vector<uint32_t> index(oracle.size());
            int i = 0;
            for (uint32_t &v: index) {
                v = i++;
            }
            random_shuffle(index.begin(), index.end());
#pragma omp parallel for
            for (uint32_t i = 0; i < C; ++i) {
                controls[i].id = index[i];
                LinearSearch(oracle, index[i], K, &controls[i].neighbors);
            }
        }
        pcontrols->swap(controls);
    }


    class KGraphImpl: public KGraph {
    protected:
        vector<uint32_t> M;
        vector<vector<Neighbor>> graph;
        bool no_dist;   // Distance & flag information in Neighbor is not valid.

    public:
        virtual ~KGraphImpl () {
        }


        virtual void build (IndexOracle const &oracle, IndexParams const &param, IndexInfo *info);


        virtual void get_nn (uint32_t id, uint32_t *nns, float *dist, uint32_t *pM, uint32_t *pL) const {
            if (!(id < graph.size())) throw invalid_argument("id too big");
            auto const &v = graph[id];
            *pM = M[id];
            *pL = v.size();
            if (nns) {
                for (uint32_t i = 0; i < v.size(); ++i) {
                    nns[i] = v[i].id;
                }
            }
            if (dist) {
                if (no_dist) throw runtime_error("distance information is not available");
                for (uint32_t i = 0; i < v.size(); ++i) {
                    dist[i] = v[i].dist;
                }
            }
        }

    };
    class KGraphConstructor: public KGraphImpl {

        static void current_best_squared_dist(const vector<uint32_t>& idA, const vector<uint32_t>& idB, Eigen::MatrixXf& D){  // Compute squared Euclidean dist between 2 matrixes
          Eigen::internal::set_is_malloc_allowed(false);
//          // (100, na)   (100, nb)  ->
          Eigen::MatrixXf A = nodes(Eigen::all, idA); // (100, na)
          Eigen::MatrixXf B = nodes(Eigen::all, idB).transpose(); // (nb, 100)
          // get square sum
          Eigen::MatrixXf A2 = square_sums(idA, Eigen::all).transpose();  // (1, na)
          Eigen::MatrixXf B2 = square_sums(idB, Eigen::all);  // (nb, 1)

          D.noalias() = -2 * B * A;
          D.noalias() += B2 * Eigen::MatrixXf::Ones(1, A2.cols());
          D.noalias() += Eigen::MatrixXf::Ones(B2.rows(),1) * (A2);

          Eigen::internal::set_is_malloc_allowed(true);
          return;
        }

        static void squared_dist(const vector<uint32_t>& idA, const vector<uint32_t>& idB, Eigen::MatrixXf& D){  // Compute squared Euclidean dist between 2 matrixes
          Eigen::internal::set_is_malloc_allowed(false);
//          // (100, na)   (100, nb)  ->
          Eigen::MatrixXf A = nodes(Eigen::all, idA); // (100, na)
          Eigen::MatrixXf B = nodes(Eigen::all, idB).transpose(); // (nb, 100)
          // get square sum
          Eigen::MatrixXf A2 = square_sums(idA, Eigen::all).transpose();  // (1, na)
          Eigen::MatrixXf B2 = square_sums(idB, Eigen::all);  // (nb, 1)

          D.noalias() = -2 * B * A;
          D.noalias() += B2 * Eigen::MatrixXf::Ones(1, A2.cols());
          D.noalias() += Eigen::MatrixXf::Ones(B2.rows(),1) * (A2);

          Eigen::internal::set_is_malloc_allowed(true);
          return;
        }



        // The neighborhood structure maintains a pool of near neighbors of an object.
        // The neighbors are stored in the pool.  "n" (<="params.L") is the number of valid entries
        // in the pool, with the beginning "k" (<="n") entries sorted.
        struct Nhood { // neighborhood
            Lock lock;
            float radius;   // distance of interesting range
            float radiusM;
            Neighbors pool;
            uint32_t L;     // # valid items in the pool,  L + 1 <= pool.size()
            uint32_t M;     // we only join items in pool[0..M)
            bool found;     // helped found new NN in this round
            vector<uint32_t> nn_old;
            vector<uint32_t> nn_new;
            vector<uint32_t> rnn_old;
            vector<uint32_t> rnn_new;

            // only non-readonly method which is supposed to be called in parallel
            uint32_t parallel_try_insert (uint32_t id, float dist) {
                if (dist > radius) return pool.size();
                LockGuard guard(lock);
                uint32_t l = UpdateKnnList(&pool[0], L, Neighbor(id, dist));

                if (l <= L) { // inserted
//                  if(pool[l].id != (id * 2 + 1))cout<<"error!!!" <<id <<" "<<pool[l].id <<"\n";
                  pool[l].id = (id<<1)|1;
//                  if(pool[l].id != (id * 2 + 1))cout<<"error!!!" <<id <<" "<<pool[l].id <<"\n";
                  if (L + 1 == pool.size()) {
                    radius = pool[L-1].dist;
                  }
                  else {
                    ++L;
                  }
                  found = true;
                }
                return l;
            }

            uint32_t parallel_try_insert_batch(const vector<uint32_t>& id_vec, uint32_t st_id, const float * dist_mat, uint32_t my_id){
              LockGuard guard(lock);
              int siz = id_vec.size();

              for(uint32_t i = st_id; i < siz; i++){
                if(i == my_id) continue;
                float dist = dist_mat[i];

                if(dist > radius) continue;
                uint32_t id = id_vec[i];
                uint32_t l = UpdateKnnList(&pool[0], L, Neighbor(id, dist));
                if (l <= L) { // inserted
                  pool[l].id = (id << 1) | 1;
                  if (L + 1 == pool.size()) {
                    radius = pool[L - 1].dist;
                  }
                  else {
                    ++L;
                  }
                  found = true;
                }
              }
              return 0;
            }

            // join should not be conflict with insert
            template <typename C>
            void join (C callback) const {
                for (uint32_t const i: nn_new) {
                    for (uint32_t const j: nn_new) {
                        if (i < j) {
                            callback(i, j);
                        }
                    }
                    for (uint32_t j: nn_old) {
                        callback(i, j);
                    }
                }
            }
        };

        IndexOracle const &oracle;
        IndexParams params;
        IndexInfo *pinfo;
        vector<Nhood> nhoods;
        size_t n_comps;

        void init () {
            uint32_t N = oracle.size();
            cout<<"N is "<<N<<"\n";
            uint32_t seed = params.seed;
            // [new] Preprocess squared sum
            cout<<"square sums start to init!\n";

            cout<<"square sums init finished!\n";
            cout<<"square sums.shape "<<square_sums.rows() <<" "<<square_sums.cols()<<"\n";
            cout<<"if row-major: "<<square_sums.IsRowMajor<<"\n";
            mt19937 rng(seed);
            for (auto &nhood: nhoods) {
//                nhood.nn_new.resize(params.S * 2);
//                nhood.nn_new.resize(params.S );
                nhood.nn_new.resize(60);        // TODO: change back to params.S
                nhood.pool.resize(params.L+1);
                nhood.radius = numeric_limits<float>::max();
            }

          boost::timer::cpu_timer timer;
#pragma omp parallel
            {
#ifdef _OPENMP
                mt19937 rng(seed ^ omp_get_thread_num());
#else
                mt19937 rng(seed);
#endif
                vector<uint32_t> random(params.S + 1);
#pragma omp  for simd
                for (uint32_t n = 0; n < N; ++n) {
                    auto &nhood = nhoods[n];
                    Neighbors &pool = nhood.pool;
                    GenRandom(rng, &nhood.nn_new[0], nhood.nn_new.size(), N);
                    GenRandom(rng, &random[0], random.size(), N);
                    nhood.L = params.S;
                    nhood.M = params.S;
                    uint32_t i = 0;

                    // Compute squared dist with Eigen
//                    Eigen::MatrixXf D;
//                    squared_dist(vector<uint32_t>({n}), random, D);
//                    const float * D_data = D.data();
//                    for (uint32_t l = 0; l < nhood.L; ++l) {
//                      if (random[i] == n) ++i;
//                      auto &nn = nhood.pool[l];
//                      nn.id = random[i];
////                        nn.dist = oracle(nn.id, n);
////                        nn.dist = D(i, 0);
//                      nn.dist = D_data[i];
//                      nn.id = (nn.id<<1)|1;
////                      nn.flag = true;
//                      ++i;
//                    }

                  // Compute squared dist with for-loop
                    for (uint32_t l = 0; l < nhood.L; ++l) {
                        if (random[i] == n) ++i;
                        auto &nn = nhood.pool[l];
                        nn.id = random[i++];
                        nn.dist = oracle(nn.id, n);
                        nn.id = (nn.id<<1)|1;
//                        nn.flag = true;
                    }
                    sort(pool.begin(), pool.begin() + nhood.L);
                }
            }
          auto times = timer.elapsed();
          cerr << "Init:  time: " << times.wall / 1e9<<"\n";

        }
        void join () {
            uint32_t total_found=0;
            size_t cc = 0;
            double total_dist_time = 0;
            double total_insert_time = 0;
//            #pragma omp parallel for default(shared) schedule(dynamic, 100) reduction(+:cc)
#ifdef SPECIFIC_TIME
            #pragma omp parallel for default(shared) schedule(dynamic, 100) reduction(+:total_dist_time) reduction(+:total_insert_time)
#else
  #ifdef DIST_CNT
              #pragma omp parallel for default(shared) schedule(dynamic, 100) reduction(+:cc)
  #else
              #pragma omp parallel for simd default(shared) schedule(dynamic, 100)
//              #pragma omp parallel for simd default(shared) schedule(static, 100)
  #endif
#endif
            for (uint32_t n = 0; n < oracle.size(); ++n) {
#ifdef DIST_CNT
                size_t uu = 0;
#endif
////                // Eigen-version
////                // step 1. Compute all dist
                boost::timer::cpu_timer timer;

                nhoods[n].nn_old.insert(nhoods[n].nn_old.begin(), nhoods[n].nn_new.begin(), nhoods[n].nn_new.end() );

                const vector<uint32_t>& nn_new = nhoods[n].nn_new;
                const vector<uint32_t>& nn_old = nhoods[n].nn_old;

                Eigen::MatrixXf D(nn_old.size(), nn_new.size());
                squared_dist(nn_new, nn_old, D);



#ifdef SPECIFIC_TIME
                auto times = timer.elapsed();
//                cerr << "compute dist:  time: " << times.wall / 1e9<<"\n";
                total_dist_time += (times.wall / 1e9);
#endif
              float* data = D.data();
              uint32_t cols = D.cols(), rows = D.rows();
              size_t data_id = 0;


              // Ground Truth for reference
//              for(size_t ia = 0; ia < cols;ia++){
//                uint32_t i = nn_new[ia];
////                for(size_t ib = ia + 1; ib < rows; ib++){
//                for(size_t ib = 0; ib < rows; ib++){
//                  if(ib == ia)continue;
//                  uint32_t j = nn_old[ib];
//                  float dis = D(ib, ia);
//                  nhoods[i].parallel_try_insert(j, dis);
//                }
//              }

              // Batch insert
              for(size_t ia = 0; ia < cols; ia++, data_id += rows) {
                uint32_t i = nn_new[ia];
                nhoods[i].parallel_try_insert_batch(nn_old, 0, data + data_id, ia);
              }
//              if(rows > cols) {

//                Eigen::internal::set_is_malloc_allowed(false);
                Eigen::MatrixXf B = D.bottomRows(rows - cols).transpose();
//                D = D.bottomRows(rows - cols).transpose();


//                D = D.bottomRows(rows - cols);
//                D.transposeInPlace();
//                Eigen::internal::set_is_malloc_allowed(true);

                // Ground Truth for reference
//              for(size_t ib=cols; ib < rows; ib++){
//                uint32_t j = nn_old[ib];
//                for(size_t ia = 0; ia<cols;ia++){
//                  uint32_t i = nn_new[ia];
//                  float dis = D(ia, ib);
//                  nhoods[j].parallel_try_insert(i, dis);
//                }
//              }

                float *dataT = B.data();
//              data_id = cols * cols;
                data_id = 0;
                for (size_t ib = cols; ib < rows; ib++, data_id += cols) {
                  uint32_t j = nn_old[ib];
                  nhoods[j].parallel_try_insert_batch(nn_new, 0, dataT + data_id, 10000);
                }
//              }



              vector<uint32_t>().swap(nhoods[n].nn_new);
              vector<uint32_t>().swap(nhoods[n].nn_old);

#ifdef SPECIFIC_TIME
                auto new_times = timer.elapsed();
                total_insert_time += (new_times.wall - times.wall) / 1e9;
#endif
#ifdef DIST_CNT
                total_found = total_found+ (uu > 0);
#endif
            }
#ifdef SPECIFIC_TIME
            cout<<"Total dist time is "<<total_dist_time<<"\n";
            cout<<"Total insert time is "<<total_insert_time<<"\n";
#endif
#ifdef DIST_CNT
            n_comps += cc;
            printf("cc: %lu\n",cc);
            printf("total_found: %u\n",total_found);
#endif
        }


        void update () {
            uint32_t N = oracle.size();

            // compute new M and corresponding radius2
//            #pragma omp parallel for
            #pragma omp simd
//            #pragma omp parallel for simd
            for (uint32_t n = 0; n < N; ++n) {
                auto &nhood = nhoods[n];
                if (nhood.found) {
                    uint32_t maxl = std::min(nhood.M + params.S, nhood.L);
                    uint32_t c = 0;
                    uint32_t l = 0;

                    while ((l < maxl) && (c < params.S)) {
//                        if (nhood.pool[l].flag) ++c;
//                        if (nhood.flags.test(l)) ++c;
                        if (nhood.pool[l].id & 1) ++c;
                        ++l;
                    }
                    nhood.M = l;
                }
                BOOST_VERIFY(nhood.M > 0);
                nhood.radiusM = nhood.pool[nhood.M-1].dist;
            }
            // Reset the variables corresponding to the nhoods to be updated
            #pragma omp parallel for simd
            for (uint32_t n = 0; n < N; ++n) {
#ifdef _OPENMP
              mt19937 rng(params.seed ^ omp_get_thread_num());
#else
              mt19937 rng(params.seed);
#endif

                auto &nhood = nhoods[n];
                nhood.found = false;
                auto &nn_new = nhood.nn_new;
                auto &nn_old = nhood.nn_old;
                for (uint32_t l = 0; l < nhood.M; ++l) {
                    auto &nn = nhood.pool[l];
                    uint32_t nn_real_id = nn.id >> 1;
//                    if(nn_real_id >=N)cout<<"Errroor!!\n";
                    auto &nhood_o = nhoods[nn_real_id];  // nhood on the other side of the edge
//                    if (nn.flag) {
//                    if (nhood.flags.test(l)) {
                    if (nn.id & 1) {
                        nn_new.push_back(nn_real_id);
                        if (nn.dist > nhood_o.radiusM) {
                            LockGuard guard(nhood_o.lock);
                            if(nhood_o.rnn_new.size() < params.R)
                              nhood_o.rnn_new.push_back(n);
                            else{
                              uint32_t loc = rng() % params.R;
//                              uint32_t past_M = nhoods[nhood_o.rnn_new[loc]].M;
//                              uint32_t new_M  = nhood.M;
//                              if(new_M < past_M)
                                nhood_o.rnn_new[loc] = n;
                            }
                        }
//                        nn.id &= (~(uint32_t)1);
                        nn.id &= 4294967294u;

                    }
                    else {
//                        nn_old.push_back(nn.id);
                        nn_old.push_back(nn_real_id);
                        if (nn.dist > nhood_o.radiusM) {
                            LockGuard guard(nhood_o.lock);
//                            nhood_o.rnn_old.push_back(n);
                            if(nhood_o.rnn_old.size() < params.R)
                              nhood_o.rnn_old.push_back(n);
                            else{
                              uint32_t loc = rng() % params.R;
//                              uint32_t past_M = nhoods[nhood_o.rnn_old[loc]].M;
//                              uint32_t new_M  = nhood.M;
//                              if(new_M < past_M)
                                nhood_o.rnn_old[loc] = n;
                            }
                        }
                    }
                }
            }
            // sample #params.R nodes from new & old of 【rnn】
            // not sure if better
#pragma omp simd
#pragma unroll
            for (uint32_t i = 0; i < N; ++i) {
                auto &nn_new = nhoods[i].nn_new;
                auto &nn_old = nhoods[i].nn_old;
                auto &rnn_new = nhoods[i].rnn_new;
                auto &rnn_old = nhoods[i].rnn_old;

                nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
                nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
                vector<uint32_t>().swap(rnn_new);
                vector<uint32_t>().swap(rnn_old);
            }
#ifdef SHOW_MEM_SIZE
            uint32_t nhood_size        = sizeof(nhoods[0]);
            uint32_t nhood_pool_size   = sizeof(nhoods[0].pool[0]) * nhoods[0].pool.size();
            uint32_t nhood_nn_new_size = sizeof(nhoods[0].nn_new[0]) * nhoods[0].nn_new.size();
            uint32_t nhood_nn_old_size = sizeof(nhoods[0].nn_old[0]) * nhoods[0].nn_old.size();
            double total_estimate = 1.0 * (nhood_size + nhood_pool_size + nhood_nn_new_size + nhood_nn_old_size) * N / 1024 /1024 /1024;
          // help function
            cout<<"sizeof(nhoods[0]) " << nhood_size <<"\n";
            cout<<"sizeof(nhoods[0].pool) " << nhood_pool_size <<"\n";
            cout<<"sizeof(nhoods[0].nn_new) " << nhood_nn_new_size <<"\n";
            cout<<"sizeof(nhoods[0].nn_old) " << nhood_nn_old_size <<"\n";
            cout<<"Estimate total size is ["<<total_estimate<<"]\n\n";
# endif
        }


public:
        KGraphConstructor (IndexOracle const &o, IndexParams const &p, IndexInfo *r)
            : oracle(o), params(p), pinfo(r), nhoods(o.size()), n_comps(0)
        {
            no_dist = false;
            boost::timer::cpu_timer timer;

            uint32_t N = oracle.size();

            square_sums = nodes.colwise().squaredNorm();
            vector<Control> controls;
            if (verbosity > 0) cerr << "Generating control..." << endl;
            GenerateControl(oracle, params.controls, params.K, &controls);
            if (verbosity > 0) cerr << "Initializing..." << endl;
            // initialize nhoods
            init();

          // iterate until converge
            float total = N * float(N - 1) / 2;
            IndexInfo info;
            info.stop_condition = IndexInfo::ITERATION;
            info.recall = 0;
            info.accuracy = numeric_limits<float>::max();
            info.cost = 0;
            info.iterations = 0;
            info.delta = 1.0;

            for (uint32_t it = 0; (params.iterations <= 0) || (it < params.iterations); ++it) {
                // start the clock for this itr
                auto start_itr = chrono::high_resolution_clock::now();
                ++info.iterations;
                join();
              auto stop_join = chrono::high_resolution_clock::now();


              if(!params.if_eval) {
                auto times = timer.elapsed();
                cerr << "iteration: " << info.iterations<< " time: " << times.wall / 1e9<<"\n";
              }
              else {
                // 【start the clock for this itr】
                {
                  info.cost = n_comps / total;
                  accumulator_set<float, stats<tag::mean>> one_exact;
                  accumulator_set<float, stats<tag::mean>> one_approx;
                  accumulator_set<float, stats<tag::mean>> one_recall;
                  accumulator_set<float, stats<tag::mean>> recall;
                  accumulator_set<float, stats<tag::mean>> accuracy;
                  accumulator_set<float, stats<tag::mean>> M;
                  accumulator_set<float, stats<tag::mean>> delta;
                  for (auto const &nhood: nhoods) {
                    M(nhood.M);
                    delta(EvaluateDelta(nhood.pool, params.K));
                  }
                  for (auto const &c: controls) {
                    one_approx(nhoods[c.id].pool[0].dist);
                    one_exact(c.neighbors[0].dist);
                    one_recall(EvaluateOneRecall(nhoods[c.id].pool, c.neighbors));
                    recall(EvaluateRecall(nhoods[c.id].pool, c.neighbors));
                    accuracy(EvaluateAccuracy(nhoods[c.id].pool, c.neighbors));
                  }
                  info.delta = mean(delta);
                  info.recall = mean(recall);
                  info.accuracy = mean(accuracy);
                  info.M = mean(M);
                  auto times = timer.elapsed();
                  if (verbosity > 0) {
                    cerr << "iteration: " << info.iterations
                         << " recall: " << info.recall
                         << " accuracy: " << info.accuracy
                         << " cost: " << info.cost
                         << " M: " << info.M
                         << " delta: " << info.delta
                         << " time: " << times.wall / 1e9
                         << " one-recall: " << mean(one_recall)
                         << " one-ratio: " << mean(one_approx) / mean(one_exact)
                         << endl;
                  }
                }
                if (info.delta <= params.delta) {
                  info.stop_condition = IndexInfo::DELTA;
                  break;
                }
                if (info.recall >= params.recall) {
                  info.stop_condition = IndexInfo::RECALL;
                  break;
                }
              }
                if(it < params.iterations -1) { // not last loop
                  update();
                }
                // start the clock
                auto stop_update = chrono::high_resolution_clock::now();
                // compute the elapsed time
                auto elapsed_time = chrono::duration_cast<chrono::duration<double>>(stop_join - start_itr);
                cout << "Join    elapsed: " << elapsed_time.count() << " seconds\n";

                elapsed_time = chrono::duration_cast<chrono::duration<double>>(stop_update - stop_join);
                cout << "Update  elapsed: " << elapsed_time.count() << " seconds\n";
            }
            // Save id to knn
            knng.resize(N);
            for (uint32_t n = 0; n < N; ++n) {
              auto &knn = knng[n];
              auto const &pool = nhoods[n].pool;
              uint32_t K = 100;
              knn.resize(K);

              uint32_t  i = 0;

              for (uint32_t k = 0; k < K; ++k, ++i) {
                uint32_t pool_i_id = pool[i].id >> 1;
                if(pool_i_id == n) ++i;
                knn[k] = pool_i_id;
              }
            }

        }

    };

    void KGraphImpl::build (IndexOracle const &oracle, IndexParams const &param, IndexInfo *info) {
        KGraphConstructor con(oracle, param, info);
        knng.swap(con.knng);
    }

    KGraph *KGraph::create () {
        return new KGraphImpl;
    }
}

