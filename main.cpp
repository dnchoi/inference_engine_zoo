#include "iostream"
#include "frvf_onnx.h"
#include "configparser.h"
#include <unistd.h>
#include <spdlog/spdlog.h> 
#include "spdlog/sinks/basic_file_sink.h"
#include <assert.h>
#include <fstream>

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
    spdlog::set_level(spdlog::level::debug); 

    // auto file_logger = spdlog::basic_logger_mt("main", "logs/log.log");
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
        char * _model;
        char * _engine;
        char * _file;
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
    SPDLOG_INFO("img path : {}", args._file);
    std::string _engine(args._engine);
    if(_engine == "onnx"){

        try
        {
            onnx_frvf::frvf_onnx *onnx;
            onnx = new onnx_frvf::frvf_onnx(args._model, args._acc, args._opti, args._B, args._C, args._W, args._H, args._file);
            std::vector<float> avg_ms(0, args._iter);

            for(int num = 0; num < args._iter; num++){
                float pro_time = onnx->do_inference();
                avg_ms.push_back(pro_time);
                SPDLOG_INFO("Iter number : {} / Processing time : {:03.8f}", num, pro_time);
            }
            float average = std::accumulate( avg_ms.begin(), avg_ms.end(), 0.0 ) / avg_ms.size();

            SPDLOG_INFO("{:03.8f} micro / {:03.8f} milli", average, average/1000);

        	std::string filePath = "output.csv";

            std::ofstream writeFile(filePath.data());
            if( writeFile.is_open() ){
                writeFile << "BATCH,CHANNEL,WIDTH,HEIGHT,ITERATE,ACCELERATOR,OPTIMIZER,MODEL,ENGINE,FILE,MICRO,MILLI\n";
                writeFile << args._B << "," << args._C << "," << args._W << "," 
                          << args._H << "," << args._iter << "," << args._acc << "," 
                          << args._opti << "," << args._model << "," << args._engine << "," 
                          << args._file << "," << average << "," << average/1000 << "\n";
                
                writeFile.close(); 
            }
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