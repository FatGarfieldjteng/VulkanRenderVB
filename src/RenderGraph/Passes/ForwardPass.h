#pragma once

#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderGraph.h"
#include "Scene/ECS.h"
#include "Scene/Scene.h"

#include <volk.h>
#include <glm/glm.hpp>
#include <vector>

struct GPUMesh;
class DescriptorManager;

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
        const std::vector<GPUMesh>*         gpuMeshes;
        const std::vector<GPUMaterialData>* gpuMaterials;
    };

    explicit ForwardPass(const Desc& desc);

    void Setup(RenderGraph& graph, PassHandle self) override;
    void Execute(VkCommandBuffer cmd) override;

private:
    Desc mDesc;
};
