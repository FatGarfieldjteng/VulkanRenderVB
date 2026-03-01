#pragma once

#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderGraph.h"

class PresentPass : public RenderPass {
public:
    struct Desc {
        ResourceHandle swapchainResource;
        PassHandle     forwardPassHandle;
    };

    explicit PresentPass(const Desc& desc);

    void Setup(RenderGraph& graph, PassHandle self) override;
    void Execute(VkCommandBuffer cmd) override;

private:
    Desc mDesc;
};
