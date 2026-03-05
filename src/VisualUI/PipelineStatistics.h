#pragma once

#include <volk.h>
#include <cstdint>
#include <vector>

class PipelineStatistics {
public:
    void Initialize(VkDevice device, uint32_t framesInFlight);
    void Shutdown(VkDevice device);

    void BeginPass(VkCommandBuffer cmd, uint32_t frameIndex);
    void EndPass(VkCommandBuffer cmd, uint32_t frameIndex);
    void CollectResults(VkDevice device, uint32_t frameIndex);

    struct Stats {
        uint64_t vertexShaderInvocations   = 0;
        uint64_t fragmentShaderInvocations = 0;
        uint64_t computeShaderInvocations  = 0;
        uint64_t clippingPrimitives        = 0;
    };

    const Stats& GetStats() const { return mLatest; }
    bool IsEnabled() const { return mEnabled; }
    void SetEnabled(bool e) { mEnabled = e; }

private:
    struct FrameData {
        VkQueryPool pool   = VK_NULL_HANDLE;
        bool        active = false;
    };

    std::vector<FrameData> mFrames;
    Stats                  mLatest{};
    bool                   mEnabled = false;
};
