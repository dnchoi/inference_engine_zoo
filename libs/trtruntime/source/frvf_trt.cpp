#include <frvf_trt.h>

using namespace trt_frvf;
static yolo_Logger gLogger;

frvf_trt::frvf_trt(const char * file_path, bool useCUDA, 
				   int OPT_OPTION, int B, int C, int W, int H)
{
	this->_Instance(file_path, useCUDA, OPT_OPTION, B, C, W, H);
}

void frvf_trt::_Instance(const char * file_path, bool useCUDA, 
						 int OPT_OPTION, int B, int C, int W, int H)
{
	SPDLOG_INFO(file_path);
    SPDLOG_INFO(useCUDA);
    SPDLOG_INFO(OPT_OPTION);
    SPDLOG_INFO(B);
    SPDLOG_INFO(C);
    SPDLOG_INFO(W);
    SPDLOG_INFO(H);
    input_B = B;
    input_C = C;
    input_W = W;
    input_H = H;
    modelName = (std::string)file_path;
    int index;
    for(int i=0; i <modelName.length(); i++){
        if(modelName[i]=='.'){ //found index
            index = i;
        }
    }

    trtName = modelName.substr(0,index)+".trt";

    this->parseOnnxModel();
    this->loadModel();
}

nvinfer1::ICudaEngine* frvf_trt::createEngine(){
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::ICudaEngine* _engine = nullptr;
    // parse ONNX
    if (!parser->parseFromFile(modelName.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return _engine;
    }
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // we have only one image in batch
    builder->setMaxBatchSize(input_B);
    
    auto profile = builder->createOptimizationProfile();
    // profile->setDimensions("inputs", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{B, C, 320, 320});
    // profile->setDimensions("inputs", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{B, C, H, W});
    // profile->setDimensions("inputs", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{B, C, 640, 640});
    config->addOptimizationProfile(profile);
    // auto pro = builder->createBuilderConfig();
    // auto preprocessorConfig = makeUnique(builder->createNetworkConfig());
    // generate TensorRT engine optimized for the target platform
    // engine->reset(builder->buildEngineWithConfig(*network, *config));
    _engine = builder->buildEngineWithConfig(*network, *config);
    builder->destroy();
    config->destroy();
    network->destroy();
    
    return _engine;
}

nvinfer1::IHostMemory * frvf_trt::createModelStream(nvinfer1::ICudaEngine *_engine){
    nvinfer1::IHostMemory* _m_stream{ nullptr };

    SPDLOG_ERROR("A");
    _m_stream = _engine->serialize();
    assert(_m_stream != nullptr);
    _engine->destroy();

    return _m_stream;
}

void frvf_trt::parseOnnxModel()
{
    SPDLOG_INFO(trtName);
    std::ifstream ifile;
    
    ifile.open(trtName);
    SPDLOG_INFO(ifile.good());
    if(!ifile.good()){
        nvinfer1::ICudaEngine *_engine = nullptr;
        nvinfer1::IHostMemory* modelStream{ nullptr };

        SPDLOG_INFO("Create engine file Doing");
        _engine = createEngine();
        if(_engine == nullptr) SPDLOG_CRITICAL("Engine file is nullptr, please check engine loop");
        else SPDLOG_INFO("Create engine file");

        modelStream = createModelStream(_engine);
        std::ofstream p(trtName, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        
        modelStream->destroy();
        SPDLOG_ERROR("A");
        SPDLOG_INFO("Create engine file Done");
        SPDLOG_ERROR(trtName);
    }
    else if(ifile.good()){
        SPDLOG_INFO("Engine file exists");
    }
}

void frvf_trt::loadModel(){
    std::ifstream file(trtName, std::ios::binary);
    if (!file.good()) {
        SPDLOG_CRITICAL("Read {} error!", trtName);
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    runtime = nvinfer1::createInferRuntime(gLogger);
    if(runtime == nullptr){ SPDLOG_CRITICAL("runtime is nullptr"); }
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    if(engine == nullptr){ SPDLOG_CRITICAL("engine is nullptr"); }
    context = engine->createExecutionContext();
    if(context == nullptr){ SPDLOG_CRITICAL("context is nullptr"); }
    delete[] trtModelStream;
    // assert(engine->getNbBindings() == 2);

    total_node_number = engine->getNbBindings();
    SPDLOG_INFO("This model input / output node is : {}", total_node_number);
    int nodeIndex[total_node_number];
    nodeDims = new nvinfer1::Dims[total_node_number]; 

    for(int a=0; a<total_node_number; a++){
        nodeDims[a] = engine->getBindingDimensions(engine->getBindingIndex(engine->getBindingName(a)));
        SPDLOG_INFO("node Index : {}\tnode Name : {}\tnodeDims[{}] : {} {} {} {} {}", engine->getBindingIndex(engine->getBindingName(a)), engine->getBindingName(a), a, nodeDims[a].d[0], nodeDims[a].d[1], nodeDims[a].d[2], nodeDims[a].d[3], nodeDims[a].d[4]);
    }
    SPDLOG_INFO(nodeDims[0].d[0] * nodeDims[0].d[1] * nodeDims[0].d[2] * nodeDims[0].d[3]);
    CUDA_CHECK(cudaStreamCreate(&stream));
    SPDLOG_INFO("Get model engine");
    SPDLOG_INFO("> {}", nodeDims[0].d[0] * nodeDims[0].d[1] * nodeDims[0].d[2] * nodeDims[0].d[3]);

    data = new float[nodeDims[0].d[0] * nodeDims[0].d[1] * nodeDims[0].d[2] * nodeDims[0].d[3]];
    prob = new float[nodeDims[1].d[0] * nodeDims[1].d[1]];

}

float frvf_trt::do_inference(cv::Mat frame)
{
    try
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        void * buffer[total_node_number];

        CUDA_CHECK(cudaMalloc(&buffer[0], nodeDims[0].d[0] * nodeDims[0].d[1] * nodeDims[0].d[2] * nodeDims[0].d[3] * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffer[1], nodeDims[1].d[0] * nodeDims[1].d[1] * sizeof(float)));

        preprocess_img(data, frame, input_W, input_H); // letterbox BGR to RGB

        CUDA_CHECK(cudaMemcpyAsync(buffer[0], data, nodeDims[0].d[0] * nodeDims[0].d[1] * nodeDims[0].d[2] * nodeDims[0].d[3] * sizeof(float), cudaMemcpyHostToDevice, stream));
        context->enqueue(nodeDims[0].d[0], buffer, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(prob, buffer[1], nodeDims[1].d[0] * nodeDims[1].d[1] * nodeDims[1].d[2] * nodeDims[1].d[3] * nodeDims[1].d[4] * sizeof(float), cudaMemcpyDeviceToHost, stream));

        
        cudaFree(*buffer);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    	return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    }
    catch(const std::exception& e)
    
    {
        std::cerr << e.what() << '\n';
    }
    
}

frvf_trt::~frvf_trt()
{
    SPDLOG_INFO("TensorRT class destroy");
    engine->destroy();
    context->destroy();
}