#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <functional>
#include <chrono>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>

#include <ThreadPool.h>

#include <cuda_runtime.h>

class TorchAotOp
{
private:
    const std::string _modelFilePath;
    typedef std::shared_ptr<torch::inductor::AOTIModelContainerRunner> AOTIModelContainerRunnerPtr;
    AOTIModelContainerRunnerPtr _aotiModelContainerRunner;

public:
    TorchAotOp(const std::string& modelFilePath): _modelFilePath(modelFilePath)
    {
        _aotiModelContainerRunner = std::make_shared<torch::inductor::AOTIModelContainerRunnerCuda>(modelFilePath, 40);
    }

    void compute(const std::vector<torch::Tensor>& const_inputs, std::vector<torch::Tensor>& outputs) {
        std::vector<torch::Tensor> inputs;
        for (int i = 0; i < const_inputs.size(); ++i)
        {   
            auto input = const_inputs[i].to(torch::kCUDA);
            // inputs.push_back(const_inputs[i].to(torch::kCUDA));
            inputs.push_back(input);
        }
        outputs = _aotiModelContainerRunner->run(inputs);
    }
};

int main()
{
    // init the operator
    // TorchAotOp torchAotOp("../../fx_experiments/dynamicLib/user_model_trim_innerwei_cuda_fp32.so");
    TorchAotOp torchAotOp("./dynamicLib_7622_gpu/user_model_trim_innerwei_cuda_fp32.so");

    // 初始化线程池 init the thread_pool
    threadpool::ThreadPool threadPool(8, 12, 8);

    // inputs ready
    std::vector<torch::Tensor> concat_4_s;
    for (int i = 0; i < 1000; ++i) {
        concat_4_s.push_back(torch::rand({i % 256 + 1, 688}));
    }
    std::vector<torch::Tensor> concat_6_user_tile_s;
    for (int i = 0; i < 1000; ++i) {
        concat_6_user_tile_s.push_back(torch::rand({i % 256 + 1, 50, 176}));
    }
    std::vector<torch::Tensor> concat_8_user_tile_s;
    for (int i = 0; i < 1000; ++i) {
        concat_8_user_tile_s.push_back(torch::rand({i % 256 + 1, 1}));
    }
    std::vector<torch::Tensor> concat_7_user_tile_s;
    for (int i = 0; i < 1000; ++i) {
        concat_7_user_tile_s.push_back(torch::rand({i % 256 + 1, 50, 172}));
    }
    std::vector<torch::Tensor> concat_9_user_tile_s;
    for (int i = 0; i < 1000; ++i) {
        concat_9_user_tile_s.push_back(torch::rand({i % 256 + 1, 1}));
    }
    std::vector<torch::Tensor> concat_user_tile_s;
    for (int i = 0; i < 1000; ++i) {
        concat_user_tile_s.push_back(torch::rand({i % 256 + 1, 4}));
    }
    std::vector<torch::Tensor> concat_5_s;
    for (int i = 0; i < 1000; ++i) {
        concat_5_s.push_back(torch::rand({i % 256 + 1, 156}));
    }
    std::vector<torch::Tensor> concat_2_user_tile_s;
    for (int i = 0; i < 1000; ++i) {
        concat_2_user_tile_s.push_back(torch::rand({i % 256 + 1, 16}));
    }
    std::vector<torch::Tensor> concat_1_user_tile_s;
    for (int i = 0; i < 1000; ++i) {
        concat_1_user_tile_s.push_back(torch::rand({i % 256 + 1, 204}));
    }

    for (int cnt = 0; cnt < 1000; cnt++) {
        std::vector<torch::Tensor> inputs;
        inputs.push_back(concat_4_s[cnt % 1000]);
        inputs.push_back(concat_6_user_tile_s[cnt % 1000]);
        inputs.push_back(concat_8_user_tile_s[cnt % 1000]);
        inputs.push_back(concat_7_user_tile_s[cnt % 1000]);
        inputs.push_back(concat_9_user_tile_s[cnt % 1000]);
        inputs.push_back(concat_user_tile_s[cnt % 1000]);
        inputs.push_back(concat_5_s[cnt % 1000]);
        inputs.push_back(concat_2_user_tile_s[cnt % 1000]);
        inputs.push_back(concat_1_user_tile_s[cnt % 1000]);

        auto fn = [&, cnt](std::vector<torch::Tensor> inputs, /*TorchAotOp* torchAotOpPtr,*/ void* arg) {
            auto torchAotOpPtr = &torchAotOp;
            // start time
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<torch::Tensor> outputs;
            // torchAotOpPtr->compute(inputs, outputs);
            torchAotOp.compute(inputs, outputs);
            // end time
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            printf("iter: %d duration: %lf ms \n", cnt, (double)duration / 1000.0);
        };

        std::function<void (void*)> task = std::bind(fn, inputs, /*&torchAotOp,*/ std::placeholders::_1);

        threadPool.AddTask(task, nullptr);

    }
    return 0;
}