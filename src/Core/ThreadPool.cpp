#include "Core/ThreadPool.h"
#include "Core/Logger.h"

void ThreadPool::Initialize(uint32_t numThreads) {
    if (numThreads == 0)
        numThreads = std::max(1u, std::thread::hardware_concurrency() - 1);

    for (uint32_t i = 0; i < numThreads; i++) {
        mWorkers.emplace_back([this] {
            for (;;) {
                std::packaged_task<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mQueueMutex);
                    mCondition.wait(lock, [this] { return mStopping || !mTasks.empty(); });
                    if (mStopping && mTasks.empty()) return;
                    task = std::move(mTasks.front());
                    mTasks.pop();
                }
                task();
                {
                    std::lock_guard<std::mutex> lock(mQueueMutex);
                    mActiveTasks--;
                }
                mFinishedCondition.notify_one();
            }
        });
    }
    LOG_INFO("ThreadPool initialized with {} workers", numThreads);
}

void ThreadPool::Shutdown() {
    {
        std::lock_guard<std::mutex> lock(mQueueMutex);
        mStopping = true;
    }
    mCondition.notify_all();
    for (auto& w : mWorkers) {
        if (w.joinable()) w.join();
    }
    mWorkers.clear();
    mStopping = false;
    LOG_INFO("ThreadPool shut down");
}

std::future<void> ThreadPool::Submit(std::function<void()> task) {
    std::packaged_task<void()> pt(std::move(task));
    auto future = pt.get_future();
    {
        std::lock_guard<std::mutex> lock(mQueueMutex);
        mActiveTasks++;
        mTasks.push(std::move(pt));
    }
    mCondition.notify_one();
    return future;
}

void ThreadPool::WaitAll() {
    std::unique_lock<std::mutex> lock(mQueueMutex);
    mFinishedCondition.wait(lock, [this] { return mActiveTasks == 0 && mTasks.empty(); });
}
