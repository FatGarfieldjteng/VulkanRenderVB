#pragma once

#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderGraph.h"
#include "RayTracing/AccelStructure.h"
#include "RayTracing/RTShadows.h"
#include "RayTracing/RTReflections.h"

#include <volk.h>
#include <glm/glm.hpp>

class RayTracingPass : public RenderPass {
public:
    struct Desc {
        ResourceHandle depthResource;
        ResourceHandle colorResource;     // HDR color to composite into
        PassHandle     forwardPassHandle;

        RTShadows*      shadows    = nullptr;
        RTReflections*  reflections = nullptr;
        AccelStructure* accel       = nullptr;

        VkImageView depthView   = VK_NULL_HANDLE;
        VkSampler   depthSampler = VK_NULL_HANDLE;
        VkExtent2D  extent;

        glm::mat4 invViewProj;
        glm::vec3 lightDir;
        float     lightRadius;
        glm::vec3 cameraPos;
        float     roughness;

        // Composite
        VkPipeline       compositePipeline   = VK_NULL_HANDLE;
        VkPipelineLayout compositePipeLayout = VK_NULL_HANDLE;
        VkDescriptorSet  compositeDescSet    = VK_NULL_HANDLE;

        float shadowStrength     = 1.0f;
        float reflectionStrength = 0.3f;
        bool  debugShadowVis    = false;
    };

    explicit RayTracingPass(const Desc& desc);

    void Setup(RenderGraph& graph, PassHandle self) override;
    void Execute(VkCommandBuffer cmd) override;

private:
    Desc mDesc;
};
