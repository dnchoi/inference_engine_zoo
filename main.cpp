#include "iostream"
#include "frvf_onnx.h"
#include <spdlog/spdlog.h>
#include "configparser.h"
#include <unistd.h>
#include "spdlog/sinks/basic_file_sink.h"
#include <assert.h>

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO


int main(int argc, char* argv[]){
    /**
     * @brief Construct a new frvf onnx::frvf onnx object and get Instence
     * @arg model_path, useCUDA, optimizer
     * @param model_path type string 
     * @param useCUDA type bool 
     * @param optimizer type int 
     * 0 = ORT_DISABLE_ALL : To disable all optimizations
     * 1 = ORT_ENABLE_BASIC : To enable basic optimizations (Such as redundant node removals) 
     * 2 = ORT_ENABLE_EXTENDED : To enable extended optimizations(Includes level 1 + more complex optimizations like node fusions)
     * 3 = ORT_ENABLE_ALL : To Enable All possible optimizations
     */
    
    // auto file_logger = spdlog::basic_logger_mt("basic_logger", "logs/basic.txt");
    // spdlog::set_default_logger(file_logger);

    struct Arg
    {
        int _B;
        int _W;
        int _H;
        int _C;
        int _iter;
        int _acc;
        int _opti;
        std::string _model;
        std::string _engine;
        std::string _file;
    };
    
    Arg args;
    CConfigParser config("config.ini");
#ifdef _CDN_DEBUG_
	if (config.IsSuccess()) {
        args._B = config.GetInt("B");
        args._W = config.GetInt("W");
        args._H = config.GetInt("H");
        args._C = config.GetInt("C");
        args._iter = config.GetInt("ITERATION");
        args._acc = config.GetInt("ACCELERATOR");
        args._opti = config.GetInt("OPTIMIZER");
        args._model = config.GetString("MODEL");
        args._engine = config.GetString("ENGINE");
        args._file = config.GetString("FILE");	
    }
#elif _DEBUG_
        args._B = atoi(argv[1]);
        args._C = atoi(argv[2]);
        args._W = atoi(argv[3]);
        args._H = atoi(argv[4]);
        args._iter = atoi(argv[5]);
        args._acc = atoi(argv[6]);
        args._opti = atoi(argv[7]);
        args._model = argv[8];
        args._engine = argv[9];
        args._file = argv[10];
#else
#endif
    SPDLOG_INFO("batch size : {}", args._B);
    SPDLOG_INFO("input channel : {}", args._C);
    SPDLOG_INFO("input width : {}", args._W);
    SPDLOG_INFO("input height : {}", args._H);
    SPDLOG_INFO("iteration number : {}", args._iter);
    SPDLOG_INFO("accelerator : {}", args._acc);
    SPDLOG_INFO("optimizer : {}", args._opti);
    SPDLOG_INFO("model path : {}", args._model);
    SPDLOG_INFO("engine : {}", args._engine);
    SPDLOG_INFO("file name : {}", args._file);
    if(args._engine == "onnx"){
        onnx_frvf::frvf_onnx *onnx;
        onnx = new onnx_frvf::frvf_onnx(args._model, args._acc, args._opti);

        try
        {
            float avg_ms = 0.0;
            // avg_ms = onnx->do_inference(args._B, args._C, args._W, args._H, "img.png");
            // SPDLOG_INFO("Pokemon img : {:03.8f}", avg_ms);
            // sleep(5);
            
            // avg_ms = 0.0;
            std::vector<float> process_vec = onnx->workspace(args._iter, args._B, args._C, args._W, args._H, args._file);
            for(int q = 0; q < process_vec.size(); q++)
            {
                avg_ms += process_vec[q];
            }
            avg_ms = avg_ms / 1000.0;
            SPDLOG_INFO("{} : {:03.8f}", args._file, avg_ms);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
            
    }
    else{
        return 0;
    }
	return 0;
}