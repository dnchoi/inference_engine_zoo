#include "iostream"
#include "frvf_onnx.h"
#include <spdlog/spdlog.h>
#include "configparser.h"

// #include "spdlog/sinks/basic_file_sink.h"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG


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
    };
    
    Arg args;
    CConfigParser config("test.ini");
	if (config.IsSuccess()) {
        args._B = config.GetInt("B");
        args._W = config.GetInt("W");
        args._H = config.GetInt("H");
        args._C = config.GetInt("C");
        args._iter = config.GetInt("ITERATION");
        args._acc = config.GetInt("ACCELERATOR");

        args._model = config.GetString("MODEL");
        args._engine = config.GetString("ENGINE");
	}
    
    SPDLOG_INFO("batch size : {}", args._B);
    SPDLOG_INFO("input width : {}", args._W);
    SPDLOG_INFO("input height : {}", args._H);
    SPDLOG_INFO("input channel : {}", args._C);
    SPDLOG_INFO("iteration number : {}", args._iter);
    SPDLOG_INFO("accelerator : {}", args._acc);
    SPDLOG_INFO("model path : {}", args._model);
    SPDLOG_INFO("optimizer : {}", args._engine);
    if(args._engine == "onnx"){
        onnx_frvf::frvf_onnx *onnx;
        onnx = new onnx_frvf::frvf_onnx(rgs._model, true, 0);
        std::vector<float> result;
        float avg_ms = 0.0;
        for(int i = 0; i < 1000; i++){
            result.push_back(onnx->do_inference("img.png"));
        }
        for(int q = 0; q < result.size(); q++)
        {
            avg_ms += result[q];
        }
        avg_ms = avg_ms / 1000.0;
        SPDLOG_CRITICAL("{:03.8f}", avg_ms);
    }
    else{
        return 0;
    }
	return 0;
}