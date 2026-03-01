#pragma once

#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderGraph.h"
#include "Lighting/CascadedShadowMap.h"
#include "Scene/ECS.h"

#include <volk.h>
#include <glm/glm.hpp>
#include <vector>

struct GPUMesh;

class ShadowPass : public RenderPass {
public:
    struct Desc {
        ResourceHandle         csmResource;
        const CascadedShadowMap* csm;
        VkPipeline             pipeline;
        VkPipelineLayout       pipelineLayout;
        const Registry*        registry;
        const std::vector<GPUMesh>* gpuMeshes;
    };

    explicit ShadowPass(const Desc& desc);

    void Setup(RenderGraph& graph, PassHandle self) override;
    void Execute(VkCommandBuffer cmd) override;

private:
    Desc mDesc;
    PassHandle mSelf = UINT32_MAX;
};
