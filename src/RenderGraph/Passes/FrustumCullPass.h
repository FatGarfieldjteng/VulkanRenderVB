#pragma once

#include "RenderGraph/RenderPass.h"

class ComputeCulling;
struct CullParams;

class FrustumCullPass : public RenderPass {
public:
    struct Desc {
        const ComputeCulling* culling = nullptr;
        const CullParams*     params  = nullptr;
    };

    explicit FrustumCullPass(const Desc& desc);

    void Setup(RenderGraph& graph, PassHandle self) override;
    void Execute(VkCommandBuffer cmd) override;

private:
    Desc mDesc;
};
