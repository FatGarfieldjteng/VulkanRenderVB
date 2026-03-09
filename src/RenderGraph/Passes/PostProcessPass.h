#pragma once

#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderGraph.h"

class PostProcessStack;

class PostProcessPass : public RenderPass {
public:
    using ResourceHandle = RenderPass::ResourceHandle;
    using PassHandle     = RenderPass::PassHandle;

    struct Desc {
        ResourceHandle       hdrResource;
        ResourceHandle       depthResource;
        ResourceHandle       swapchainResource;
        PassHandle           forwardPassHandle;
        PostProcessStack*    stack;
        VkImageView          swapchainView;
        VkImageView          depthView;
        VkExtent2D           extent;
        float                deltaTime;
        const float*         invProjection;
        const float*         projInfo;
        float                farPlane;
    };

    explicit PostProcessPass(const Desc& desc);

    void Setup(RenderGraph& graph, PassHandle self) override;
    void Execute(VkCommandBuffer cmd) override;

private:
    Desc mDesc;
};
