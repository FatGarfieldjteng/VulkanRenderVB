#pragma once

#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderGraph.h"
#include "RayTracing/PathTracer.h"
#include "RayTracing/NRDDenoiser.h"

#include <volk.h>
#include <glm/glm.hpp>

class PathTracingPass : public RenderPass {
public:
    struct Desc {
        ResourceHandle colorResource;
        PassHandle     forwardPassHandle = UINT32_MAX;

        PathTracer*    pathTracer  = nullptr;
        NRDDenoiser*   denoiser    = nullptr;

        VkExtent2D extent;
        glm::mat4  invViewProj;
        glm::mat4  viewProj;
        glm::mat4  viewMat;   // worldToView (for NRD)
        glm::mat4  projMat;   // viewToClip (for NRD)
        glm::mat4  viewMatPrev;
        glm::mat4  projMatPrev;
        glm::vec3  cameraPos;
        glm::vec3  sunDir;
        glm::vec3  sunColor;
        float      sunIntensity;
        float      lightRadius;

        bool enableDenoiser = true;

        // Composite: blit denoised (or raw) output to HDR image
        VkPipeline       compositePipeline   = VK_NULL_HANDLE;
        VkPipelineLayout compositePipeLayout = VK_NULL_HANDLE;
        VkDescriptorSet  compositeDescSet    = VK_NULL_HANDLE;

        // Split-screen: pixel columns [splitLeft, splitRight) receive PT output
        uint32_t splitLeft  = 0;
        uint32_t splitRight = 0;  // 0 means use full width

        // Compare mode: split-screen L=denoised, R=raw (compareSplitX = pixel x of split)
        uint32_t compareSplitX = 0;  // 0 = not in compare mode
    };

    explicit PathTracingPass(const Desc& desc);

    void Setup(RenderGraph& graph, PassHandle self) override;
    void Execute(VkCommandBuffer cmd) override;

private:
    Desc mDesc;
};
