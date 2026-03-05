#pragma once

#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderGraph.h"

class DebugUI;

class ImGuiPass : public RenderPass {
public:
    using ResourceHandle = RenderPass::ResourceHandle;
    using PassHandle     = RenderPass::PassHandle;

    struct Desc {
        ResourceHandle swapchainResource;
        PassHandle     previousPassHandle;
        VkImageView    swapchainView;
        VkExtent2D     extent;
        DebugUI*       debugUI;
    };

    explicit ImGuiPass(const Desc& desc);

    void Setup(RenderGraph& graph, PassHandle self) override;
    void Execute(VkCommandBuffer cmd) override;

private:
    Desc mDesc;
};
