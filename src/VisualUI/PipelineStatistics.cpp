#include "VisualUI/PipelineStatistics.h"
#include "Core/Logger.h"

#include <cstring>

static constexpr VkQueryPipelineStatisticFlags kStatFlags =
    VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
    VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT |
    VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT |
    VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT;

static constexpr uint32_t kStatCount = 4;

void PipelineStatistics::Initialize(VkDevice device, uint32_t framesInFlight) {
    mFrames.resize(framesInFlight);

    for (auto& f : mFrames) {
        VkQueryPoolCreateInfo ci{};
        ci.sType              = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        ci.queryType          = VK_QUERY_TYPE_PIPELINE_STATISTICS;
        ci.queryCount         = 1;
        ci.pipelineStatistics = kStatFlags;
        vkCreateQueryPool(device, &ci, nullptr, &f.pool);
    }

    LOG_INFO("PipelineStatistics initialized ({} frames)", framesInFlight);
}

void PipelineStatistics::Shutdown(VkDevice device) {
    for (auto& f : mFrames) {
        if (f.pool) {
            vkDestroyQueryPool(device, f.pool, nullptr);
            f.pool = VK_NULL_HANDLE;
        }
    }
    mFrames.clear();
}

void PipelineStatistics::BeginPass(VkCommandBuffer cmd, uint32_t frameIndex) {
    if (!mEnabled) return;
    auto& f = mFrames[frameIndex];
    vkCmdResetQueryPool(cmd, f.pool, 0, 1);
    vkCmdBeginQuery(cmd, f.pool, 0, 0);
    f.active = true;
}

void PipelineStatistics::EndPass(VkCommandBuffer cmd, uint32_t frameIndex) {
    if (!mEnabled) return;
    auto& f = mFrames[frameIndex];
    if (!f.active) return;
    vkCmdEndQuery(cmd, f.pool, 0);
    f.active = false;
}

void PipelineStatistics::CollectResults(VkDevice device, uint32_t frameIndex) {
    if (!mEnabled) return;
    auto& f = mFrames[frameIndex];

    uint64_t data[kStatCount]{};
    VkResult res = vkGetQueryPoolResults(
        device, f.pool, 0, 1,
        sizeof(data), data, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (res == VK_SUCCESS) {
        mLatest.vertexShaderInvocations   = data[0];
        mLatest.fragmentShaderInvocations = data[1];
        mLatest.computeShaderInvocations  = data[2];
        mLatest.clippingPrimitives        = data[3];
    }
}
