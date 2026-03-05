#pragma once

#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderGraph.h"

class ComputeCulling;
struct CullParams;

class OcclusionTestPass : public RenderPass {
public:
    using PassHandle = RenderPass::PassHandle;

    using ResourceHandle = RenderPass::ResourceHandle;

    struct Desc {
        PassHandle            hiZBuildPassHandle;
        ResourceHandle        depthResource;
        const ComputeCulling* culling;
        const CullParams*     params;
    };

    explicit OcclusionTestPass(const Desc& desc);

    void Setup(RenderGraph& graph, PassHandle self) override;
    void Execute(VkCommandBuffer cmd) override;

private:
    Desc mDesc;
};
