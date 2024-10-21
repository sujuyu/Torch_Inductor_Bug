The dynamic library obtained by torch._export.aot_compile will cause the smallest recurrence of the illegal memory access problem when running in parallel with multiple threads.


1. compile
need to modify the TORCH_DIR; e.g. cmake -DTORCH_DIR=/home/admin/zy429782/miniforge3/envs/torch231_cuda121/lib/python3.8/site-packages/torch
```bash
mkdir build &&  cd  build && cmake -DTORCH_DIR=/your/custom/path/to/torch  ../ && make
```
need to modify the TORCH_DIR; e.g. 
```bash
cmake -DTORCH_DIR=/home/admin/zy429782/miniforge3/envs/torch231_cuda121/lib/python3.8/site-packages/torch
```

2. run
```bash
cd ../ && ./build/threadpool_torchaot
```
Maybe you need to run it a few more times. You will get the error info like this:
```
iter: 289 duration: 26.697000 ms 
iter: 288 duration: 32.372000 ms 
iter: 291 duration: 30.422000 ms 
Error: CUDA driver error: an illegal memory access was encountered
terminate called after throwing an instance of 'std::runtime_error'
  what():  run_func_( container_handle_, input_handles.data(), input_handles.size(), output_handles.data(), output_handles.size(), cuda_stream_handle, proxy_executor_handle_) API call failed at ../torch/csrc/inductor/aoti_runner/model_container_runner.cpp, line 75
Aborted (core dumped)
```
