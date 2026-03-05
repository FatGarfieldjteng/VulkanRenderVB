#include "VisualUI/GPUProfiler.h"
#include "Core/Logger.h"

#include <fstream>
#include <cstring>
#include <algorithm>
#include <numeric>

void GPUProfiler::Initialize(VkDevice device, VkPhysicalDevice physDevice,
                              uint32_t framesInFlight, uint32_t maxScopes) {
    mMaxScopes = maxScopes;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physDevice, &props);
    mTimestampPeriod = props.limits.timestampPeriod;

    mFrames.resize(framesInFlight);
    uint32_t queryCount = maxScopes * 2 + 2;

    for (auto& f : mFrames) {
        VkQueryPoolCreateInfo ci{};
        ci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        ci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        ci.queryCount = queryCount;
        vkCreateQueryPool(device, &ci, nullptr, &f.queryPool);
        f.queryCount = queryCount;
        f.scopeNames.reserve(maxScopes);
    }

    LOG_INFO("GPUProfiler initialized ({} scopes, {:.2f} ns/tick)", maxScopes, mTimestampPeriod);
}

void GPUProfiler::Shutdown(VkDevice device) {
    for (auto& f : mFrames) {
        if (f.queryPool) {
            vkDestroyQueryPool(device, f.queryPool, nullptr);
            f.queryPool = VK_NULL_HANDLE;
        }
    }
    mFrames.clear();
}

void GPUProfiler::BeginFrame(VkCommandBuffer cmd, uint32_t frameIndex) {
    auto& f = mFrames[frameIndex];
    vkCmdResetQueryPool(cmd, f.queryPool, 0, f.queryCount);
    f.nextQuery = 0;
    f.scopeNames.clear();
    f.scopeStack.clear();

    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, f.queryPool, f.nextQuery++);
}

void GPUProfiler::EndFrame(VkCommandBuffer cmd, uint32_t frameIndex) {
    auto& f = mFrames[frameIndex];
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, f.queryPool, f.nextQuery++);
}

void GPUProfiler::BeginScope(VkCommandBuffer cmd, uint32_t frameIndex, const char* name) {
    auto& f = mFrames[frameIndex];
    if (f.nextQuery + 2 > f.queryCount) return;

    f.scopeNames.emplace_back(name);
    f.scopeStack.push_back(f.nextQuery);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, f.queryPool, f.nextQuery++);
}

void GPUProfiler::EndScope(VkCommandBuffer cmd, uint32_t frameIndex) {
    auto& f = mFrames[frameIndex];
    if (f.scopeStack.empty()) return;
    f.scopeStack.pop_back();
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, f.queryPool, f.nextQuery++);
}

void GPUProfiler::CollectResults(VkDevice device, uint32_t frameIndex) {
    auto& f = mFrames[frameIndex];
    if (f.nextQuery == 0) return;

    std::vector<uint64_t> timestamps(f.nextQuery);
    VkResult res = vkGetQueryPoolResults(
        device, f.queryPool, 0, f.nextQuery,
        timestamps.size() * sizeof(uint64_t), timestamps.data(),
        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (res != VK_SUCCESS) return;

    mResults.clear();
    uint32_t scopeIdx = 0;
    for (uint32_t i = 1; i + 1 < f.nextQuery; i += 2) {
        if (scopeIdx >= f.scopeNames.size()) break;
        float ms = static_cast<float>(timestamps[i + 1] - timestamps[i]) * mTimestampPeriod / 1e6f;
        mResults.push_back({ f.scopeNames[scopeIdx], ms });
        scopeIdx++;
    }

    if (mHistory.size() >= kMaxHistory)
        mHistory.erase(mHistory.begin());
    mHistory.push_back({ mResults });
}

float GPUProfiler::GetTotalMs() const {
    float total = 0.0f;
    for (const auto& r : mResults) total += r.durationMs;
    return total;
}

void GPUProfiler::ExportCSV(const char* path) const {
    std::ofstream out(path);
    if (!out.is_open()) {
        LOG_ERROR("Failed to open {} for CSV export", path);
        return;
    }

    if (!mHistory.empty()) {
        out << "Frame";
        for (const auto& r : mHistory.front().results)
            out << "," << r.name;
        out << "\n";

        for (size_t f = 0; f < mHistory.size(); f++) {
            out << f;
            for (const auto& r : mHistory[f].results)
                out << "," << r.durationMs;
            out << "\n";
        }
    }

    LOG_INFO("GPU profiler exported {} frames to {}", mHistory.size(), path);
}

void GPUProfiler::ExportChromeTracing(const char* path) const {
    std::ofstream out(path);
    if (!out.is_open()) {
        LOG_ERROR("Failed to open {} for Chrome tracing export", path);
        return;
    }

    out << "[";
    bool first = true;
    for (size_t f = 0; f < mHistory.size(); f++) {
        float offset = 0.0f;
        for (const auto& r : mHistory[f].results) {
            if (!first) out << ",";
            first = false;
            float tsUs = static_cast<float>(f) * 16666.0f + offset * 1000.0f;
            float durUs = r.durationMs * 1000.0f;
            out << "{\"name\":\"" << r.name << "\","
                << "\"cat\":\"gpu\","
                << "\"ph\":\"X\","
                << "\"ts\":" << tsUs << ","
                << "\"dur\":" << durUs << ","
                << "\"pid\":0,\"tid\":0}";
            offset += r.durationMs;
        }
    }
    out << "]";

    LOG_INFO("GPU profiler Chrome tracing exported {} frames to {}", mHistory.size(), path);
}
