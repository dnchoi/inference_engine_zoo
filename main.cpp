#include "iostream"
#include "frvf_onnx.h"
#include "configparser.h"
#include <unistd.h>
#include <spdlog/spdlog.h> 
#include "spdlog/sinks/basic_file_sink.h"
#include <assert.h>
#include <fstream>
#include <filesystem>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

bool DirectoryExists( const char* pzPath )
{
    if ( pzPath == NULL) return false;

    DIR *pDir;
    bool bExists = false;

    pDir = opendir (pzPath);

    if (pDir != NULL)
    {
        bExists = true;    
        (void) closedir (pDir);
    }

    return bExists;
}

std::vector<std::string> split(std::string input, char delimiter) {
    std::vector<std::string> answer;
    std::stringstream ss(input);
    std::string temp;
 
    while (getline(ss, temp, delimiter)) {
        answer.push_back(temp);
    }
 
    return answer;
}

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
    char * output_folder = "output";
    
    if (!std::filesystem::exists(output_folder)){
        SPDLOG_WARN("Create Directory");
        int nResult = mkdir(output_folder, 0777);
        if( nResult == 0 )
        {
            SPDLOG_INFO( "Created Directory" );
        }
        else if( nResult == -1 )
        {
            SPDLOG_ERROR( "Failed Directory Create\n" );
            SPDLOG_ERROR( "errorno : {}", errno );
        }
    }
    else{
        SPDLOG_INFO("Directory exists");
    }

    if(_engine == "onnx"){
        try
        {
            if(DirectoryExists(args._model)){
                SPDLOG_WARN("Directory mode");
                for (const auto & file : std::filesystem::directory_iterator(args._model)){
                    std::vector<float> avg_ms(0, args._iter-1);
                    const char *_model_name = file.path().c_str();
                    onnx_frvf::frvf_onnx *onnx;
                    onnx = new onnx_frvf::frvf_onnx(_model_name, args._acc, args._opti, args._B, args._C, args._W, args._H, args._file);
                    std::vector<std::string> result = split((std::string)_model_name, '/');
                    std::string filePath = (std::string)output_folder + "/" + result[result.size()-1]+".csv";
                    SPDLOG_INFO(filePath);
                    std::ofstream writeFile(filePath);
                    if( writeFile.is_open() ){
                        writeFile << "processing time,";
                        for(int num = 0; num < args._iter; num++){
                            float pro_time = onnx->do_inference();
                            if(num != 0){
                                writeFile << pro_time << ",";
                                avg_ms.push_back(pro_time);
                                SPDLOG_INFO("Iter number : {} / Processing time : {:03.8f}", num, pro_time);
                            }
                        }
                        SPDLOG_INFO(avg_ms.size());
                        float average = std::accumulate( avg_ms.begin(), avg_ms.end(), 0.0 ) / avg_ms.size();
                        writeFile << "\naverage time," << average << "\n";
                        SPDLOG_INFO("{:03.8f} micro / {:03.8f} milli", average, average/1000);
                            
                    }
                    writeFile.close(); 
                }
            }
            else{
                SPDLOG_WARN("File mode");
                std::vector<float> avg_ms(0, args._iter-1);
                onnx_frvf::frvf_onnx *onnx;
                onnx = new onnx_frvf::frvf_onnx(args._model, args._acc, args._opti, args._B, args._C, args._W, args._H, args._file);
                std::vector<std::string> result = split((std::string)args._model, '/');
                std::string filePath = (std::string)output_folder + "/" + result[result.size()-1]+".csv";
                SPDLOG_INFO(filePath);
                std::ofstream writeFile(filePath);
                if( writeFile.is_open() ){
                    writeFile << "processing time,";
                    for(int num = 0; num < args._iter; num++){
                        float pro_time = onnx->do_inference();
                        if(num != 0){
                            writeFile << pro_time << ",";
                            avg_ms.push_back(pro_time);
                            SPDLOG_INFO("Iter number : {} / Processing time : {:03.8f}", num, pro_time);
                        }
                    }
                    SPDLOG_INFO(avg_ms.size());
                    float average = std::accumulate( avg_ms.begin(), avg_ms.end(), 0.0 ) / avg_ms.size();
                    writeFile << "\naverage time," << average << "\n";
                    SPDLOG_INFO("{:03.8f} micro / {:03.8f} milli", average, average/1000);
                        
                }
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