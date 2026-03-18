#pragma once

#include <volk.h>
#include <vector>
#include <cstdint>

class RTPipeline {
public:
    uint32_t AddStage(VkShaderStageFlagBits stage, VkShaderModule module,
                      const char* entryPoint = "main");

    uint32_t AddRayGenGroup(uint32_t generalShaderIdx);
    uint32_t AddMissGroup(uint32_t generalShaderIdx);
    uint32_t AddHitGroup(uint32_t closestHitIdx,
                         uint32_t anyHitIdx       = VK_SHADER_UNUSED_KHR,
                         uint32_t intersectionIdx = VK_SHADER_UNUSED_KHR);
    uint32_t AddCallableGroup(uint32_t callableIdx);

    void Build(VkDevice device, VkPipelineLayout layout,
               uint32_t maxRecursionDepth = 1);
    void Destroy(VkDevice device);

    VkPipeline GetPipeline() const { return mPipeline; }
    uint32_t   GetGroupCount() const { return static_cast<uint32_t>(mGroups.size()); }

    std::vector<uint8_t> GetShaderGroupHandles(VkDevice device,
                                                uint32_t handleSize) const;

private:
    std::vector<VkPipelineShaderStageCreateInfo>          mStages;
    std::vector<VkRayTracingShaderGroupCreateInfoKHR>     mGroups;
    VkPipeline mPipeline = VK_NULL_HANDLE;
};
