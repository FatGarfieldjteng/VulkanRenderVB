#pragma once

#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderGraph.h"

#include <volk.h>

class ComputeCulling;
class MeshPool;

class OccluderDepthPass : public RenderPass {
public:
    using ResourceHandle = RenderPass::ResourceHandle;
    using PassHandle     = RenderPass::PassHandle;

    struct Desc {
        ResourceHandle          depthResource;
        PassHandle              frustumCullPassHandle;
        VkExtent2D              extent;
        VkImageView             depthView;
        VkPipeline              pipeline;
        VkPipelineLayout        pipelineLayout;
        VkDescriptorSet         bindlessSet;
        VkDescriptorSet         frameDescSet;
        const MeshPool*         meshPool;
        const ComputeCulling*   culling;
        uint32_t                maxDrawCount;
    };

    explicit OccluderDepthPass(const Desc& desc);

    void Setup(RenderGraph& graph, PassHandle self) override;
    void Execute(VkCommandBuffer cmd) override;

private:
    Desc mDesc;
};
