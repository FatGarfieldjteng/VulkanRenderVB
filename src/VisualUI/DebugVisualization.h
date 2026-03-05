#pragma once

#include "VisualUI/DebugUI.h"
#include <volk.h>

class ShaderManager;
class PipelineManager;

class DebugVisualization {
public:
    void Initialize(VkDevice device, ShaderManager& shaders, PipelineManager& pipelines,
                    VkFormat colorFormat, VkDescriptorSetLayout bindlessLayout,
                    VkDescriptorSetLayout frameLayout);
    void Shutdown(VkDevice device);

    VkPipeline       GetPipeline(DebugUIState::VisMode mode) const;
    VkPipelineLayout GetPipelineLayout() const { return mPipelineLayout; }

    bool RequiresSpecialRendering(DebugUIState::VisMode mode) const;

private:
    void CreateDebugPipeline(VkDevice device, ShaderManager& shaders, PipelineManager& pipelines,
                             VkFormat colorFormat,
                             VkDescriptorSetLayout bindlessLayout,
                             VkDescriptorSetLayout frameLayout);

    VkPipelineLayout mPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       mWireframePipeline = VK_NULL_HANDLE;
    VkPipeline       mDebugModePipeline = VK_NULL_HANDLE;
};
