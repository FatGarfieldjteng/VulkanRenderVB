#pragma once

#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderGraph.h"

class HiZBuffer;

class HiZBuildPass : public RenderPass {
public:
    using ResourceHandle = RenderPass::ResourceHandle;
    using PassHandle     = RenderPass::PassHandle;

    struct Desc {
        ResourceHandle   depthResource;
        PassHandle       occluderDepthPassHandle;
        const HiZBuffer* hiZ;
    };

    explicit HiZBuildPass(const Desc& desc);

    void Setup(RenderGraph& graph, PassHandle self) override;
    void Execute(VkCommandBuffer cmd) override;

private:
    Desc mDesc;
};
