#pragma once

#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderGraph.h"
#include "Lighting/CascadedShadowMap.h"
#include "Scene/ECS.h"

#include <volk.h>
#include <glm/glm.hpp>
#include <vector>

class MeshPool;

class ShadowPass : public RenderPass {
public:
    struct Desc {
        ResourceHandle         csmResource;
        const CascadedShadowMap* csm;
        VkPipeline             pipeline;
        VkPipelineLayout       pipelineLayout;
        const Registry*        registry;
        const MeshPool*        meshPool                   = nullptr;

        bool                   skip                       = false;
        bool                   gpuDriven                  = false;
        VkPipeline             indirectPipeline           = VK_NULL_HANDLE;
        VkPipelineLayout       indirectPipelineLayout     = VK_NULL_HANDLE;
        VkDescriptorSet        indirectDescSet            = VK_NULL_HANDLE;
        VkBuffer               indirectBuffer             = VK_NULL_HANDLE;
        VkBuffer               countBuffer                = VK_NULL_HANDLE;
        uint32_t               maxDrawCount               = 0;
    };

    explicit ShadowPass(const Desc& desc);

    void Setup(RenderGraph& graph, PassHandle self) override;
    void Execute(VkCommandBuffer cmd) override;

private:
    Desc mDesc;
    PassHandle mSelf = UINT32_MAX;
};
