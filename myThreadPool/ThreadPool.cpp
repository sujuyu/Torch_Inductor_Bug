#include <assert.h>
#include <mutex>
#include <stdio.h>
#include "ThreadPool.h"

using namespace threadpool;

const int ThreadPool::INIT_NUM;
const int ThreadPool::MAX_NUM;
const int ThreadPool::MAX_IDLE_TIME_SECOND;

ThreadPool::ThreadPool(int initNum, int maxNum, int idleSec)
    : _initNum(initNum)
    , _maxNum(maxNum)
    , _idleSec(idleSec)
    , _stop(false)
    , _busyCount(0)
    , _threadCount(0)
{
    Init();
}

ThreadPool::ThreadPool()
    : _initNum(ThreadPool::INIT_NUM)
    , _maxNum(ThreadPool::MAX_NUM)
    , _idleSec(ThreadPool::MAX_IDLE_TIME_SECOND)
    , _stop(false)
    , _busyCount(0)
    , _threadCount(0)
{
    Init();
}

void ThreadPool::Init()
{
    if(_idleSec <= 0) {
        _idleSec = ThreadPool::MAX_IDLE_TIME_SECOND;
    }
    _pool.reserve(_maxNum);
    for(int i = 0; i < _maxNum; ++i) {
        if (i < _initNum) {
            _pool.push_back(new std::thread(&ThreadPool::ThreadRoutine, this, i));
        } else {
            _pool.push_back(nullptr);
        }
    }

    _threadCount.store(_initNum);
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _stop = true;
    }

    _cond.notify_all();
    for(auto& t : _pool) {
        if(t && t->joinable()) {
            t->join();
            delete t;
        }
    }
}

void ThreadPool::AddTask(const TaskType& task, void* arg) {
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _taskQueue.emplace(task, arg);
    }
    _cond.notify_all();

    PoolGrow();
}

void ThreadPool::PoolGrow()
{
    int busy = _busyCount.load();
    int threadCount = _threadCount.load();
    printf("busy: %d, threadCount: %d\n", busy, threadCount);
    if (threadCount == busy) {
        if (threadCount < _maxNum) {
            for (int i = 0; i < _maxNum; ++i) {
                if (_pool[i] == nullptr) {
                    _pool[i] = new std::thread(&ThreadPool::ThreadRoutine, this, i);
                    printf("add thread %d\n", i);
                    _threadCount.fetch_add(1);
                    break;
                }
            }
        }
    }
}

void ThreadPool::ThreadRoutine(int index)
{
    while(1) {
        std::pair<TaskType, void*> task(nullptr, 0);
        {
            std::cv_status waitStatus = std::cv_status::no_timeout;
            std::unique_lock<std::mutex> lock(_mutex);
            while(_taskQueue.empty() && waitStatus != std::cv_status::timeout && !_stop) {
                int idleTime = MAX_IDLE_TIME_SECOND;
                waitStatus = _cond.wait_for(lock, std::chrono::seconds(_idleSec));
            }
            if (!_taskQueue.empty()) {
                task = _taskQueue.front();
                _taskQueue.pop();
            } else if (_stop) {
                break;
            } else if (waitStatus == std::cv_status::timeout) {
                if (_threadCount > _initNum) {
                    _pool[index]->detach();
                    delete _pool[index];
                    _pool[index] = nullptr;
                    --_threadCount;
                    printf("delete thread %d", index);
                    break;
                }
            }
        }

        if (task.first != nullptr) {
            ++_busyCount;
            task.first(task.second);
            --_busyCount;
        }
    }
}