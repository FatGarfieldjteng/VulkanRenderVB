#pragma once

#include <volk.h>
#include <string>
#include <vector>
#include <cstdint>

class GPUProfiler {
public:
    void Initialize(VkDevice device, VkPhysicalDevice physDevice, uint32_t framesInFlight, uint32_t maxScopes);
    void Shutdown(VkDevice device);

    void BeginFrame(VkCommandBuffer cmd, uint32_t frameIndex);
    void EndFrame(VkCommandBuffer cmd, uint32_t frameIndex);

    void BeginScope(VkCommandBuffer cmd, uint32_t frameIndex, const char* name);
    void EndScope(VkCommandBuffer cmd, uint32_t frameIndex);

    void CollectResults(VkDevice device, uint32_t frameIndex);

    struct ScopeResult {
        std::string name;
        float       durationMs;
    };
    const std::vector<ScopeResult>& GetResults() const { return mResults; }
    float GetTotalMs() const;

    void ExportCSV(const char* path) const;
    void ExportChromeTracing(const char* path) const;

private:
    struct FrameData {
        VkQueryPool queryPool    = VK_NULL_HANDLE;
        uint32_t    queryCount   = 0;
        uint32_t    nextQuery    = 0;
        std::vector<std::string> scopeNames;
        std::vector<uint32_t>    scopeStack;
    };

    std::vector<FrameData>  mFrames;
    std::vector<ScopeResult> mResults;
    float                   mTimestampPeriod = 0.0f;
    uint32_t                mMaxScopes       = 0;

    struct HistoryEntry {
        std::vector<ScopeResult> results;
    };
    std::vector<HistoryEntry> mHistory;
    static constexpr size_t   kMaxHistory = 300;
};
