#pragma once

#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <cstdint>

class ThreadPool {
public:
    void Initialize(uint32_t numThreads = 0);
    void Shutdown();

    std::future<void> Submit(std::function<void()> task);
    void WaitAll();

    uint32_t GetThreadCount() const { return static_cast<uint32_t>(mWorkers.size()); }

private:
    std::vector<std::thread>          mWorkers;
    std::queue<std::packaged_task<void()>> mTasks;

    std::mutex              mQueueMutex;
    std::condition_variable mCondition;
    std::condition_variable mFinishedCondition;
    uint32_t                mActiveTasks = 0;
    bool                    mStopping    = false;
};
