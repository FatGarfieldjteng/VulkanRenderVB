#pragma once

#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderGraph.h"
#include "RayTracing/PathTracer.h"
#include "RayTracing/NRDDenoiser.h"

#include <volk.h>
#include <glm/glm.hpp>

/// Hybrid mode: rasterize G-buffer (via Forward pass), then launch RT pipeline
/// for multi-bounce GI lighting. The forward pass writes HDR + depth.
/// This pass traces secondary rays from screen-space pixels for indirect lighting
/// and composites the result into the HDR image.
class HybridRTPass : public RenderPass {
public:
    struct Desc {
        ResourceHandle depthResource;
        ResourceHandle colorResource;
        PassHandle     forwardPassHandle;

        PathTracer*    pathTracer = nullptr;
        NRDDenoiser*   denoiser   = nullptr;

        VkExtent2D extent;
        glm::mat4  invViewProj;
        glm::mat4  viewProj;
        glm::mat4  viewMat;
        glm::mat4  projMat;
        glm::mat4  viewMatPrev;
        glm::mat4  projMatPrev;
        glm::vec3  cameraPos;
        glm::vec3  sunDir;
        glm::vec3  sunColor;
        float      sunIntensity;
        float      lightRadius;

        bool enableDenoiser = true;

        VkPipeline       compositePipeline   = VK_NULL_HANDLE;
        VkPipelineLayout compositePipeLayout = VK_NULL_HANDLE;
        VkDescriptorSet  compositeDescSet    = VK_NULL_HANDLE;

        // Compare mode: split-screen L=denoised, R=raw (compareSplitX = pixel x of split)
        uint32_t compareSplitX = 0;  // 0 = not in compare mode
    };

    explicit HybridRTPass(const Desc& desc);

    void Setup(RenderGraph& graph, PassHandle self) override;
    void Execute(VkCommandBuffer cmd) override;

private:
    Desc mDesc;
};
