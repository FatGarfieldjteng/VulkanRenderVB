#include "RayTracing/RTPipeline.h"
#include "Core/Logger.h"

#include <cstring>

uint32_t RTPipeline::AddStage(VkShaderStageFlagBits stage, VkShaderModule module,
                               const char* entryPoint) {
    VkPipelineShaderStageCreateInfo ci{};
    ci.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage  = stage;
    ci.module = module;
    ci.pName  = entryPoint;
    mStages.push_back(ci);
    return static_cast<uint32_t>(mStages.size() - 1);
}

uint32_t RTPipeline::AddRayGenGroup(uint32_t generalShaderIdx) {
    VkRayTracingShaderGroupCreateInfoKHR g{};
    g.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    g.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    g.generalShader      = generalShaderIdx;
    g.closestHitShader   = VK_SHADER_UNUSED_KHR;
    g.anyHitShader       = VK_SHADER_UNUSED_KHR;
    g.intersectionShader = VK_SHADER_UNUSED_KHR;
    mGroups.push_back(g);
    return static_cast<uint32_t>(mGroups.size() - 1);
}

uint32_t RTPipeline::AddMissGroup(uint32_t generalShaderIdx) {
    VkRayTracingShaderGroupCreateInfoKHR g{};
    g.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    g.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    g.generalShader      = generalShaderIdx;
    g.closestHitShader   = VK_SHADER_UNUSED_KHR;
    g.anyHitShader       = VK_SHADER_UNUSED_KHR;
    g.intersectionShader = VK_SHADER_UNUSED_KHR;
    mGroups.push_back(g);
    return static_cast<uint32_t>(mGroups.size() - 1);
}

uint32_t RTPipeline::AddHitGroup(uint32_t closestHitIdx, uint32_t anyHitIdx,
                                  uint32_t intersectionIdx) {
    VkRayTracingShaderGroupCreateInfoKHR g{};
    g.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    g.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    g.generalShader      = VK_SHADER_UNUSED_KHR;
    g.closestHitShader   = closestHitIdx;
    g.anyHitShader       = anyHitIdx;
    g.intersectionShader = intersectionIdx;
    mGroups.push_back(g);
    return static_cast<uint32_t>(mGroups.size() - 1);
}

uint32_t RTPipeline::AddCallableGroup(uint32_t callableIdx) {
    VkRayTracingShaderGroupCreateInfoKHR g{};
    g.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    g.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    g.generalShader      = callableIdx;
    g.closestHitShader   = VK_SHADER_UNUSED_KHR;
    g.anyHitShader       = VK_SHADER_UNUSED_KHR;
    g.intersectionShader = VK_SHADER_UNUSED_KHR;
    mGroups.push_back(g);
    return static_cast<uint32_t>(mGroups.size() - 1);
}

void RTPipeline::Build(VkDevice device, VkPipelineLayout layout,
                        uint32_t maxRecursionDepth) {
    VkRayTracingPipelineCreateInfoKHR ci{};
    ci.sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    ci.stageCount                   = static_cast<uint32_t>(mStages.size());
    ci.pStages                      = mStages.data();
    ci.groupCount                   = static_cast<uint32_t>(mGroups.size());
    ci.pGroups                      = mGroups.data();
    ci.maxPipelineRayRecursionDepth = maxRecursionDepth;
    ci.layout                       = layout;

    VK_CHECK(vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                             1, &ci, nullptr, &mPipeline));
    LOG_INFO("RT pipeline built: {} stages, {} groups, maxRecursion={}",
             mStages.size(), mGroups.size(), maxRecursionDepth);
}

void RTPipeline::Destroy(VkDevice device) {
    if (mPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, mPipeline, nullptr);
        mPipeline = VK_NULL_HANDLE;
    }
    mStages.clear();
    mGroups.clear();
}

std::vector<uint8_t> RTPipeline::GetShaderGroupHandles(VkDevice device,
                                                        uint32_t handleSize) const {
    uint32_t count = static_cast<uint32_t>(mGroups.size());
    std::vector<uint8_t> handles(count * handleSize);
    VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(device, mPipeline,
                                                   0, count,
                                                   handles.size(), handles.data()));
    return handles;
}
