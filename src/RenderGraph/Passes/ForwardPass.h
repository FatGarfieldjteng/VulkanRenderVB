#pragma once

#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderGraph.h"
#include "Scene/ECS.h"
#include "Scene/Scene.h"

#include <volk.h>
#include <glm/glm.hpp>
#include <vector>

class DescriptorManager;
class MeshPool;

class ForwardPass : public RenderPass {
public:
    struct Desc {
        ResourceHandle          csmResource;
        ResourceHandle          depthResource;
        ResourceHandle          swapchainResource;
        PassHandle              shadowPassHandle;
        VkExtent2D              extent;
        VkImageView             swapchainView;
        VkImageView             depthView;
        VkPipeline              pipeline;
        VkPipelineLayout        pipelineLayout;
        VkDescriptorSet         bindlessSet;
        VkDescriptorSet         frameDescSet;
        const Registry*         registry;
        const MeshPool*         meshPool             = nullptr;
        const std::vector<GPUMaterialData>* gpuMaterials;

        bool                    gpuDriven            = false;
        bool                    occlusionEnabled     = false;
        VkPipeline              indirectPipeline     = VK_NULL_HANDLE;
        VkPipelineLayout        indirectPipelineLayout = VK_NULL_HANDLE;

        VkBuffer                occluderBuffer       = VK_NULL_HANDLE;
        VkBuffer                occluderCountBuffer  = VK_NULL_HANDLE;
        uint32_t                maxOccluderCount     = 0;
        VkBuffer                visibleBuffer        = VK_NULL_HANDLE;
        VkBuffer                visibleCountBuffer   = VK_NULL_HANDLE;
        uint32_t                maxVisibleCount      = 0;

        PassHandle              occlusionTestPassHandle = UINT32_MAX;
        PassHandle              frustumCullPassHandle   = UINT32_MAX;
    };

    explicit ForwardPass(const Desc& desc);

    void Setup(RenderGraph& graph, PassHandle self) override;
    void Execute(VkCommandBuffer cmd) override;

private:
    Desc mDesc;
};
