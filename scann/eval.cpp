#include <omp.h>
#include <cmath>
#include <stdio.h>
#include <cstring>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <hdf5.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <unordered_map>
 
 
using namespace std;
namespace fs = std::filesystem;
 
#include "scann/scann_ops/cc/scann.h"
#include "scann/proto/scann.pb.h"
#include "cmdline.h"
 
using namespace research_scann;
size_t QUERY_TEST_COUNT = 10000;
float epsilon=1e-3;
static const double nanoSinS = 1e9;
 
cmdline::parser InitArguments(int argc, char* argv[])
{
    cmdline::parser para;
    para.add<std::string>("dataDir", 'd',
                          "data dir for base data and query data", true,
                          "/tmp/data");
    para.add<std::string>("configDir", 'c',
                          "config dir for base data and query data", true,
                          "/tmp/data");
    para.add<int>("numRuns", 'R', "number of runs for QPS", false, 1);
    para.add("help", 0, "print this message");
 
    bool ok = para.parse(argc, argv);
    if (!ok || para.exist("help")) {
        std::cerr << para.error() << std::endl;
        std::cerr << para.usage() << std::endl;
        exit(0);
    }
 
    const char* dataDir = para.get<std::string>("dataDir").c_str();
    if (access(dataDir, R_OK | W_OK) != 0) {
        printf("ERROR: Data Dir Not Exist.\n");
        exit(0);
    }
    return para;
}
 
void SplitString(const std::string inputStr, const std::string sep, std::vector<int>& outputTokens)
{
    std::string sCopy = inputStr;
    int iPosEnd = 0;
    while (true) {
        iPosEnd = sCopy.find(sep);
        if (iPosEnd == -1) {
            outputTokens.emplace_back(atoi(sCopy.c_str()));
            break;
        }
        outputTokens.emplace_back(atoi(sCopy.substr(0, iPosEnd).c_str()));
        sCopy = sCopy.substr(iPosEnd + 1);
    }
}
 
void SplitStringFloat(const std::string inputStr, const std::string sep, std::vector<float>& outputTokens)
{
    std::string sCopy = inputStr;
    int iPosEnd = 0;
    while (true) {
        iPosEnd = sCopy.find(sep);
        if (iPosEnd == -1) {
            outputTokens.emplace_back(atof(sCopy.c_str()));
            break;
        }
        outputTokens.emplace_back(atof(sCopy.substr(0, iPosEnd).c_str()));
        sCopy = sCopy.substr(iPosEnd + 1);
    }
}
 
static const char *HDF5_DATASET_TRAIN = "train";
static const char *HDF5_DATASET_TEST = "test";
static const char *HDF5_DATASET_NEIGHBORS = "neighbors";
static const char *HDF5_DATASET_DISTANCES = "distances";
 
 
void *hdf5_read(const std::string &file_name, const std::string &dataset_name, H5T_class_t dataset_class,
                int32_t &d_out, int32_t &n_out)
{
    hid_t file, dataset, datatype, dataspace, memspace;
    H5T_class_t t_class;      /* data type class */
    hsize_t dimsm[3];         /* memory space dimensions */
    hsize_t dims_out[2];      /* dataset dimensions */
    hsize_t count[2];         /* size of the hyperslab in the file */
    hsize_t offset[2];        /* hyperslab offset in the file */
    hsize_t count_out[3];     /* size of the hyperslab in memory */
    hsize_t offset_out[3];    /* hyperslab offset in memory */
    void *data_out = nullptr; /* output buffer */
 
    /* Open the file and the dataset. */
    file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen2(file, dataset_name.c_str(), H5P_DEFAULT);
 
    /* Get datatype and dataspace handles and then query
     * dataset class, order, size, rank and dimensions. */
    datatype = H5Dget_type(dataset); /* datatype handle */
    t_class = H5Tget_class(datatype);
    // assert(t_class == dataset_class || !"Illegal dataset class type");
 
    dataspace = H5Dget_space(dataset); /* dataspace handle */
    H5Sget_simple_extent_dims(dataspace, dims_out, nullptr);
    n_out = dims_out[0];
    d_out = dims_out[1];
 
    /* Define hyperslab in the dataset. */
    offset[0] = offset[1] = 0;
    count[0] = dims_out[0];
    count[1] = dims_out[1];
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
 
    /* Define the memory dataspace. */
    dimsm[0] = dims_out[0];
    dimsm[1] = dims_out[1];
    dimsm[2] = 1;
    memspace = H5Screate_simple(3, dimsm, nullptr);
 
    /* Define memory hyperslab. */
    offset_out[0] = offset_out[1] = offset_out[2] = 0;
    count_out[0] = dims_out[0];
    count_out[1] = dims_out[1];
    count_out[2] = 1;
    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, nullptr, count_out, nullptr);
 
    /* Read data from hyperslab in the file into the hyperslab in memory and display. */
    switch (t_class) {
        case H5T_INTEGER:
            data_out = new int32_t[dims_out[0] * dims_out[1]];
            H5Dread(dataset, H5T_NATIVE_INT32, memspace, dataspace, H5P_DEFAULT, data_out);  // read error
 
            break;
        case H5T_FLOAT:
            data_out = new float[dims_out[0] * dims_out[1]];
            H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, data_out);
            break;
        default:
            printf("Illegal dataset class type\n");
            break;
    }
 
    /* Close/release resources. */
    H5Tclose(datatype);
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Sclose(memspace);
    H5Fclose(file);
 
    return data_out;
}
 
void normalizeVector(float* data, int size) {
    // Calculate the norm
    float norm = 0.0;
    for (int i = 0; i < size; ++i) {
        norm += data[i] * data[i];
    }
    norm = std::sqrt(norm);
 
    // Check if the norm is zero
    if (norm == 0) {
        for (int i = 0; i < size; ++i) {
            data[i] = 1.0f / std::sqrt(size);
        }
    } else {
        // Normalize the vector
        for (int i = 0; i < size; ++i) {
            data[i] /= norm;
        }
    }
}
 
void loadHDF(const std::string &ann_file_name, int32_t &nb, int32_t &nq, int32_t &dim, int32_t &gt_closest,
    float *&data, float *&queries, int64_t *&gt_ids, float *&gt_dist, std::string metricType)
{
    data = (float *)hdf5_read(ann_file_name, HDF5_DATASET_TRAIN, H5T_FLOAT, dim, nb);
    if (metricType == "dot_product") {
        for (int i = 0; i < nb; i ++)
            normalizeVector(data + i * dim, dim);
    }
    queries = (float *)hdf5_read(ann_file_name, HDF5_DATASET_TEST, H5T_FLOAT, dim, nq);
    if (metricType == "dot_product") {
        for (int i = 0; i < nq; i ++)
            normalizeVector(queries + i * dim, dim);
    }
    int32_t *gt_ids_short = (int32_t *)hdf5_read(ann_file_name, HDF5_DATASET_NEIGHBORS, H5T_INTEGER, gt_closest, nq);
    gt_ids = new int64_t[gt_closest * nq];
    for (int i = 0; i < gt_closest * nq; i++) {
        gt_ids[i] = gt_ids_short[i];
    }
    delete[] gt_ids_short;
    float *dist_short = (float *)hdf5_read(ann_file_name, HDF5_DATASET_DISTANCES, H5T_FLOAT, gt_closest, nq);
    gt_dist = new float[gt_closest * nq];
    for (int i = 0; i < gt_closest * nq; i++) {
        gt_dist[i] = dist_short[i];
    }
    delete[] dist_short;
}
 
struct Config {
    int n_leaves;
    double avq_threshold;
    int dims_per_block;
    std::string metric_type;
    float soar_lambda;
    float overretrieve_factor;
    std::vector<int> nprobes;
    std::vector<int> reorders;
    std::vector<float> adp_thresholds;
    std::vector<int> adp_refineds;
    std::vector<int> batch_sizes;
    std::vector<int> num_threads;
    int topK;
    std::string index_save_or_load;
    std::string index_path;
    int gtK;
    int doParallel;
};
 
 
Config read_config(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cout << "[ERROR] config file not found" << std::endl;
    }
 
    Config config;
    std::unordered_map<std::string, std::string> config_map;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream is_line(line);
        std::string key;
        if (std::getline(is_line, key, '=')) {
            std::string value;
            if (std::getline(is_line, value)) {
                config_map[key] = value;
            }
        }
    }

    config.n_leaves = std::stoi(config_map["n_leaves"]);
    if (config_map["avq_threshold"] == "nan") {
        config.avq_threshold = std::numeric_limits<double>::quiet_NaN();
    } else {
        config.avq_threshold = std::stod(config_map["avq_threshold"]);
    }
    config.dims_per_block = std::stoi(config_map["dims_per_block"]);
    config.metric_type = config_map["metric_type"];
    if (config_map.find("soar_lambda") != config_map.end() && config_map.find("overretrieve_factor") != config_map.end()) {
        config.soar_lambda = std::stof(config_map["soar_lambda"]);
        config.overretrieve_factor = std::stof(config_map["overretrieve_factor"]);
    } else {
        config.soar_lambda = -1;
        config.overretrieve_factor = -1;
    }
    SplitString(config_map["nprobes"], ",", config.nprobes);
    SplitString(config_map["reorders"], ",", config.reorders);
    SplitStringFloat(config_map["adp_thresholds"], ",", config.adp_thresholds);
    SplitString(config_map["adp_refineds"], ",", config.adp_refineds);
    SplitString(config_map["batch_sizes"], ",", config.batch_sizes);
    SplitString(config_map["num_threads"], ",", config.num_threads);
    config.topK = std::stoi(config_map["topK"]);
    config.index_save_or_load = config_map["index_save_or_load"];
    config.index_path = config_map["index_path"];
    config.gtK = std::stoi(config_map["gtK"]);
    config.doParallel = std::stoi(config_map["doParallel"]);
 
    std::cout << std::endl;
    return config;
}
 
bool ReadFileToString(const std::string& filePath, std::string* content) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return false;
    }
    content->assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return true;
}
 
class ScaNNIndex
{
 
public:
    std::unique_ptr<ScannInterface> scann;
 
    std::string config = "";
 
    int training_threads = 128;

    void Init(int n_leaves, std::string metricType, int dims_per_block, float avq_threshold, float soar_lambda, float overretrieve_factor, int topK, int32_t dim, int32_t nb) {
        std::string cmd = "python create_config.py " + std::to_string(n_leaves) + " "
                                + std::to_string(nb) + " " + metricType + " "
                                + std::to_string(dims_per_block) + " " + std::to_string(avq_threshold) + " "
                                + std::to_string(dim) + " " + std::to_string(topK) + " "
                                + std::to_string(soar_lambda) + " " + std::to_string(overretrieve_factor) + " ";
        FILE* stream = popen(cmd.c_str(), "r");
        if (stream == nullptr) {
            std::cerr << "Error executing command." << std::endl;
            return;
        }
 
        this->config = "";
        char buffer[102400];
        while (fgets(buffer, sizeof(buffer), stream) != nullptr) {
            this->config += buffer ;
        }
        pclose(stream); 
        this->scann.reset(new ScannInterface());
    }
 
    void Save(const std::string &index_path, bool relative_path) {
        scann->SerializeForAll(index_path, relative_path);
    }
 
    void Load(const std::string &index_path) {
        std::string assets_pbtxt = fs::path(index_path) / "scann_assets.pbtxt";
        std::ifstream file(assets_pbtxt);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open " + assets_pbtxt);
        }
        std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        std::string configPath = index_path + "/scann_config.pb";
        std::string fileContent;
 
        if (!ReadFileToString(configPath, &fileContent)) {
            std::cerr << "Failed to read file." << std::endl;
            return ;
        }
        ScannConfig config1;
        if (!config1.ParseFromString(fileContent)) {
            std::cerr << "Failed to parse Protobuf data." << std::endl;
            return ;
        }
        auto status = scann->Initialize(config1.DebugString(), contents);
    }
    
    void Build(int32_t nb, int32_t dim, float* xb) {
        ConstSpan<float> new_dataset(xb, nb * dim);
        auto status = scann->Initialize(new_dataset, nb, this->config, this->training_threads);
    }

    void ReBuild(string config) {
        auto status = scann->RetrainAndReindex(config);
        if (!status.ok()) {
          std::cerr << "Failed to rebuild index: " << status.status() << std::endl;
          return;
        }   
    }
    
    void SingleSearch(int topK, int nprobe, int reorder, DenseDataset<float>& ptr, float adp_threshold, 
                    int adp_refined, int num_thread, int batch_size) {
        NNResultsVector* res = new NNResultsVector[ptr.size()];
        scann->SearchAdditionalParams(adp_threshold, adp_refined, nprobe);
        scann->SetNumThreads(num_thread);
        scann->SearchBatched(ptr, MutableSpan<NNResultsVector>(res, ptr.size()), topK, reorder, nprobe);
        delete[] res;
    }
 
    void Search(int32_t nq, int32_t dim, float* xq, int topK, int nprobe, int reorder, float adp_threshold, 
                    int adp_refined, int num_thread, int batch_size, std::vector<int64_t>& result) {
        NNResultsVector* res = new NNResultsVector[nq];
        result.resize(0);
        std::vector<float> queries (xq, xq + dim * nq);
        DenseDataset<float> ptr(std::move(queries), nq);
        scann->SearchAdditionalParams(adp_threshold, adp_refined, nprobe);
        scann->SetNumThreads(num_thread);
        scann->SearchBatchedParallel(ptr, MutableSpan<NNResultsVector>(res, nq), topK, reorder, nprobe, batch_size);
        
        for (int i = 0; i < nq; ++ i) {
            auto& res0 = res[i];
            for (auto& pir : res0) {
                result.emplace_back(pir.first);
            }
        }
        delete[] res;
    }
 
    void ParallelSearch(int topK, int nprobe, int reorder, float adp_threshold, int adp_refined, int num_thread, 
                    int batch_size, DenseDataset<float>& ptr) {
        NNResultsVector* res = new NNResultsVector[ptr.size()];
        scann->SearchAdditionalParams(adp_threshold, adp_refined, nprobe);
        if (num_thread != 320)
            scann->SetNumThreads(num_thread - 1);
        scann->SearchBatchedParallel(ptr, MutableSpan<NNResultsVector>(res, ptr.size()), topK, reorder, nprobe, batch_size);
        delete[] res;
    }

    void Serialize(uint8_t *&dataPtr, size_t &dataLength) {
        int result = scann->SerializeToMemory(dataPtr, dataLength);
        if (result != 0) { 
            std::cerr << "Error: Serialization failed" << std::endl;
        }
    }

    void Deserialize(uint8_t *&dataPtr, size_t &dataLength) {
        int result = scann->LoadFromMemory(dataPtr, dataLength);
        if (result != 0) { 
            std::cerr << "Error: DeSerialization failed" << std::endl;
        }
    }

    int get_num() {
        return scann->GetNum();
    }

    int get_dim() {
        return scann->GetDim();
    }
};
 
float CalRecall(std::vector<int64_t>& ids, int32_t nq, int32_t dim, int32_t gtDim, int64_t *gtAll, int a, int topK)
{
    double recall = 0.;
    
    for (auto i = 0; i < nq; ++ i) {
        auto* gt = gtAll + i * gtDim;
        std::unordered_set<int64_t> sets(ids.data() + i * topK, ids.data() + i * topK + topK);
        for (int j = 0; j < a; ++ j) {
            if (sets.find(*(gt + j)) != sets.end()) {
                recall += 1.0;
            }
        }
    }
    return recall / float(nq) / float(a);
}
 
float CalRecallDist(std::vector<float>& distance, int32_t nq, int32_t dim, int32_t gtDim, float *gtDist, int a, int topK)
{
    float recall = 0.;
    for (auto i = 0; i < nq; ++ i) {
        auto* gt = gtDist + i * gtDim;
        float threshold = gt[topK - 1] + epsilon;
        for (int j = 0; j < a; ++ j) {
            if (distance[i * topK + j] <= threshold){
                recall += 1;
            }
        }
    }
    return recall / float(nq) / float(a);
}
 
float norm(std::vector<float>& a) {
    float sum = 0.0;
    for (float value : a) {
        sum += value * value;
    }
    return std::sqrt(sum);
}
 
float dot(std::vector<float>& a, std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
 
    float result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}
 
float calDistance(std::vector<float> a, std::vector<float> b, std::string metric_type) {
    if (metric_type == "squared_l2"){
        std::vector<float> diff(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            diff[i] = a[i] - b[i];
        }
 
        return norm(diff);
    } else if (metric_type == "dot_product"){
        float a_norm = norm(a);
        float b_norm = norm(b);
        if (a_norm == 0.0 || b_norm == 0.0) {
            throw std::invalid_argument("Zero vector encountered, division by zero");
        }
        return 1.0 - dot(a, b) / (a_norm * b_norm);
    }
}
 
 
int main(int argc, char* argv[])
{
    auto args = InitArguments(argc, argv);
    std::string dataDir = args.get<std::string>("dataDir");
    std::string configDir = args.get<std::string>("configDir");
    auto numRuns = args.get<int>("numRuns");
 
    Config config = read_config(configDir);
    float *xb_;
    float *xq_;
    int64_t *gt_ids_;
    float *gt_dists_;
    int32_t nb_, nq_, dim_, gt_closest;
    loadHDF(dataDir, nb_, nq_, dim_, gt_closest, xb_, xq_, gt_ids_, gt_dists_, config.metric_type);
 
    ScaNNIndex index;
    index.Init(config.n_leaves, config.metric_type, config.dims_per_block, config.avq_threshold, config.soar_lambda, config.overretrieve_factor, config.topK, dim_, nb_);

    if (config.index_save_or_load == "load") {
        index.Load(config.index_path);
    } else if (config.index_save_or_load == "save") { 
        index.Build(nb_, dim_, xb_);
        index.Save(config.index_path, false);
    } else {
        index.Build(nb_, dim_, xb_);
        // index.Save(config.index_path, false);
        // index.Load(config.index_path);
    }

    int num = index.get_num();
    int dim = index.get_dim();

    // rebuild
    // index.ReBuild(index.config);

    // serialize
    uint8_t *dataPtr = nullptr;
    size_t dataLength = 0;
    index.Serialize(dataPtr, dataLength);
    index.Deserialize(dataPtr, dataLength);
    if (dataPtr != nullptr) {
        delete[] dataPtr;
    }

    std::vector<int64_t> ids, ids2;
    std::vector<float> distances;
    float recall ;
    for (size_t i = 0; i < config.reorders.size(); ++ i) {
        int nprobe = config.nprobes[i];
        int reorder = config.reorders[i];
        float adp_threshold = config.adp_thresholds[i];
        int adp_refined = config.adp_refineds[i];
        int num_thread = config.num_threads[i];
        int batch_size = config.batch_sizes[i];
        float QPS_avg = 0.;
        float recall = 0.;
        int topk = config.topK;
        index.Search(nq_, dim_, xq_, topk, nprobe, reorder, adp_threshold, adp_refined, num_thread, batch_size, ids);
 
        for (int i = 0; i < nq_; ++i) {
            std::vector<float> b(xq_ + i * dim_, xq_ + i * dim_ + dim_);
            for (int j = 0; j < topk; ++j){
                std::vector<float> a(xb_ + ids[i * topk + j] * dim_, xb_ + ids[i * topk + j] * dim_ + dim_);
                float distance = calDistance(a, b, config.metric_type);
                distances.emplace_back(distance);
            }
        }
 
        recall = CalRecallDist(distances, nq_, dim_, gt_closest, gt_dists_, config.gtK, topk);
        std::cout << "dist recall = " << recall << std::endl;
 
        // float idRecall = CalRecall(ids, nq_, dim_, gt_closest, gt_ids_, config.gtK, topk);
        // std::cout << "idRecall recall = " << idRecall << std::endl;
 
        std::vector<float> qps_list;
        for (int i = 0; i < numRuns; ++i) {
            std::cout << "Runing: " << i + 1 << "/" << numRuns << std::endl;
            auto t1 = std::chrono::high_resolution_clock::now();

            // expand queries
            int expand_nq = nq_;
            std::vector<float> querys;
            if (config.doParallel > 1) {
                expand_nq = nq_ * config.doParallel;
                printf("expanded %d times, got %d queries\n", config.doParallel, expand_nq);
                querys.resize(expand_nq * dim_);
                #pragma omp parallel for
                for (int i = 0; i < expand_nq; ++i) {
                    auto idx = i % nq_;
                    for (size_t j = 0; j < dim_; ++j) {
                        querys[i * dim_ + j] = xq_[idx * dim_ + j];
                    }
                }           
            } else {
                querys.assign(xq_, xq_ + nq_ * dim_); 
            }
            DenseDataset<float> ptr(std::move(querys), expand_nq);

            index.ParallelSearch(topk, nprobe, reorder, adp_threshold, adp_refined, num_thread, batch_size, ptr);
            
            auto t2 = std::chrono::high_resolution_clock::now();
            auto delta_t = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            QPS_avg = (double)expand_nq * nanoSinS / delta_t;
            
            qps_list.push_back(QPS_avg);
        }
        auto max_iter = max_element(qps_list.begin(), qps_list.end());
        std::string paramStr = std::to_string(nprobe) + "_" + std::to_string(reorder);
        // std::cout << paramStr << ":\t" << config.gtK << "@" << config.topK << ": " << recall 
        //             << "\tQPS: " << *max_iter << std::endl;
        printf("%s:\t%d@%d\t%.3f\t%.3f\n", paramStr.c_str(), config.gtK, config.topK, recall, *max_iter);
 
        return 0;
    }
 
}