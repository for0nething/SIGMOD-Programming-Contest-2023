#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_AVX512
//#define EIGEN_USE_THREADS 1
#define EIGEN_DONT_PARALLELIZE



#ifndef WDONG_KGRAPH
#define WDONG_KGRAPH

#include <stdexcept>
#include <vector>
#include <bitset>
// Eigen
#include <Eigen/Dense>

using std::vector;
using std::bitset;
namespace kgraph {
    static uint32_t const default_iterations =  30;
    static uint32_t const default_L = 100;
    static uint32_t const default_K = 25;
    static uint32_t const default_P = 100;
    static uint32_t const default_M = 0;
    static uint32_t const default_T = 1;
    static uint32_t const default_S = 10;
    static uint32_t const default_R = 100;
    static uint32_t const default_controls = 100;
    static uint32_t const default_seed = 1998;
    static float const default_delta = 0.002;
    static float const default_recall = 0.99;
    static float const default_epsilon = 1e30;
    static uint32_t const default_verbosity = 1;

    static bool const default_if_comp_hub = false;
    static bool const default_if_eval = false;
    static uint32_t const default_stale_limit = 5;


    /// Verbosity control
    /** Set verbosity = 0 to disable information output to stderr.
     */
    extern uint32_t verbosity;

    /// Index oracle
    /** The index oracle is the user-supplied plugin that computes
     * the distance between two arbitrary objects in the dataset.
     * It is used for offline k-NN graph construction.
     */
    class IndexOracle {
    public:
        /// Returns the size of the dataset.
        virtual uint32_t size () const = 0;
        /// Computes similarity
        /**
         * 0 <= i, j < size() are the index of two objects in the dataset.
         * This method return the distance between objects i and j.
         */
        virtual float operator () (uint32_t i, uint32_t j) const = 0;
        virtual float operator () (uint32_t i, uint32_t j, float DisLim) const{ return 0; };
    };

    /// Search oracle
    /** The search oracle is the user-supplied plugin that computes
     * the distance between the query and a arbitrary object in the dataset.
     * It is used for online k-NN search.
     */
    class SearchOracle {
    public:
        /// Returns the size of the dataset.
        virtual uint32_t size () const = 0;
        /// Computes similarity
        /**
         * 0 <= i < size() are the index of an objects in the dataset.
         * This method return the distance between the query and object i.
         */
        virtual float operator () (uint32_t i) const = 0;
        /// Search with brutal force.
        /**
         * Search results are guaranteed to be ranked in ascending order of distance.
         *
         * @param K Return at most K nearest neighbors.
         * @param epsilon Only returns nearest neighbors within distance epsilon.
         * @param ids Pointer to the memory where neighbor IDs are returned.
         * @param dists Pointer to the memory where distance values are returned, can be nullptr.
         */
        uint32_t search (uint32_t K, float epsilon, uint32_t *ids, float *dists = nullptr) const;
    };

    /// The KGraph index.
    /** This is an abstract base class.  Use KGraph::create to create an instance.
     */
    class KGraph {
    public:
        static Eigen::MatrixXf nodes;
        static Eigen::VectorXf square_sums;



        std::vector<std::vector<uint32_t> > knng;
        /// Indexing parameters.
        struct IndexParams {
            uint32_t iterations; 
            uint32_t L;
            uint32_t K;
            uint32_t S;
            uint32_t R;
            uint32_t controls;
            uint32_t seed;
            float delta;
            float recall;
            uint32_t stale_limit; // if a node's neighbors have been stale for `stale_limit` iterations, skip this node in the future
            bool if_comp_hub;
            bool if_eval;

            /// Construct with default values.
            IndexParams (): iterations(default_iterations), L(default_L), K(default_K), S(default_S), R(default_R), controls(default_controls), seed(default_seed), delta(default_delta), recall(default_recall),
            stale_limit(default_stale_limit), if_comp_hub(default_if_comp_hub), if_eval(default_if_eval){
            }
        };

        /// Search parameters.
        struct SearchParams {
            uint32_t K;
            uint32_t M;
            uint32_t P;
            uint32_t S;
            uint32_t T;
            float epsilon;
            uint32_t seed;
            uint32_t init;

            /// Construct with default values.
            SearchParams (): K(default_K), M(default_M), P(default_P), S(default_S), T(default_T), epsilon(default_epsilon), seed(1998), init(0) {
            }
        };

        enum {
            FORMAT_DEFAULT = 0,
            FORMAT_NO_DIST = 1,
            FORMAT_TEXT = 128
        };

        /// Information and statistics of the indexing algorithm.
        struct IndexInfo {
            enum StopCondition {
                ITERATION = 0,
                DELTA,
                RECALL
            } stop_condition;
            uint32_t iterations;
            float cost;
            float recall;
            float accuracy;
            float delta;
            float M;
        };

        /// Information and statistics of the search algorithm.
        struct SearchInfo {
            float cost;
            uint32_t updates;
        };

        virtual ~KGraph () {
        }

        /// Build the index
        virtual void build (IndexOracle const &oracle, IndexParams const &params, IndexInfo *info = 0) = 0;
        /// Constructor.
        static KGraph *create ();

        /// Get offline computed k-NNs of a given object.
        /**
         * See the full version of get_nn.
         */
        virtual void get_nn (uint32_t id, uint32_t *nns, uint32_t *M, uint32_t *L) const {
            get_nn(id, nns, nullptr, M, L);
        }
        /// Get offline computed k-NNs of a given object.
        /**
         * The user must provide space to save IndexParams::L values.
         * The actually returned L could be smaller than IndexParams::L, and
         * M <= L is the number of neighbors KGraph thinks
         * could be most useful for online search, and is usually < L.
         * If the index has been pruned, the returned L could be smaller than
         * IndexParams::L used to construct the index.
         *
         * @params id Object ID whose neighbor information are returned.
         * @params nns Neighbor IDs, must have space to save IndexParams::L values. 
         * @params dists Distance values, must have space to save IndexParams::L values.
         * @params M Useful number of neighbors, output only.
         * @params L Actually returned number of neighbors, output only.
         */
        virtual void get_nn (uint32_t id, uint32_t *nns, float *dists, uint32_t *M, uint32_t *L) const = 0;

    };
}

#if __cplusplus > 199711L
#include <functional>
namespace kgraph {
    /// Oracle adapter for datasets stored in a vector-like container.
    /**
     * If the dataset is stored in a container of CONTAINER_TYPE that supports
     * - a size() method that returns the number of objects.
     * - a [] operator that returns the const reference to an object.
     * This class can be used to provide a wrapper to facilitate creating
     * the index and search oracles.
     *
     * The user must provide a callback function that takes in two
     * const references to objects and returns a distance value.
     */
    template <typename CONTAINER_TYPE, typename OBJECT_TYPE>
    class VectorOracle: public IndexOracle {
    public:
        typedef std::function<float(OBJECT_TYPE const &, OBJECT_TYPE const &)> METRIC_TYPE;
        typedef std::function<float(OBJECT_TYPE const &, OBJECT_TYPE const &, uint32_t, uint32_t )> ID_METRIC_TYPE;
    private:
        CONTAINER_TYPE const &data;
        METRIC_TYPE dist;
    public:
        ID_METRIC_TYPE id_dist;
        class VectorSearchOracle: public SearchOracle {
            CONTAINER_TYPE const &data;
            OBJECT_TYPE const query;
            METRIC_TYPE dist;
        public:
            VectorSearchOracle (CONTAINER_TYPE const &p, OBJECT_TYPE const &q, METRIC_TYPE m): data(p), query(q), dist(m) {
            }
            virtual uint32_t size () const {   // Change for Eigen
                return data.cols();
            }
            virtual float operator () (uint32_t i) const {
                return dist(data[i], query);
            }

        };
        /// Constructor.
        /**
         * @param d: the container that holds the dataset.
         * @param m: a callback function for distance computation.  m(d[i], d[j]) must be
         *  a valid expression to compute distance.
         */
//        VectorOracle (CONTAINER_TYPE const &d, METRIC_TYPE m): data(d), dist(m) {
//        }
        VectorOracle (CONTAINER_TYPE const &d, METRIC_TYPE m, ID_METRIC_TYPE opt_m): data(d), dist(m), id_dist(opt_m) {
        }
        virtual uint32_t size () const{  // Change for Eigen
            return data.cols();
        }
//        virtual float operator () (uint32_t i, uint32_t j) const {
//            return dist(data[i], data[j]);
//        }


        virtual float operator () (uint32_t i, uint32_t j) const {  // Change for Eigen
//            return id_dist(data[i], data[j], i, j);
            return id_dist(data(Eigen::all, i), data(Eigen::all, j), i, j);
//            return dist(data.col(i), data.col(j));
        }

        /// Constructs a search oracle for query object q.
        VectorSearchOracle query (OBJECT_TYPE const &q) const {
            return VectorSearchOracle(data, q, dist);
        }
    };
    // Dummy Oracle, without oracle function
    class DummyOracle : public IndexOracle{
      public:
        uint32_t _siz;
        DummyOracle (uint32_t siz){
          _siz = siz;
        };
        /// Returns the size of the dataset.
        uint32_t size () const override{
          return _siz;
        };
        /// Computes similarity
        /**
         * 0 <= i, j < size() are the index of two objects in the dataset.
         * This method return the distance between objects i and j.
         */
        float operator () (uint32_t i, uint32_t j)const override  { return 10; };
        float operator () (uint32_t i, uint32_t j, float DisLim)const override { return 10; };
    };



    class invalid_argument: public std::invalid_argument {
    public:
        using std::invalid_argument::invalid_argument;
    };

    class runtime_error: public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    class io_error: public runtime_error {
    public:
        using runtime_error::runtime_error;
    };

}
#endif

#endif

