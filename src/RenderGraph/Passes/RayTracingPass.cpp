#include "RenderGraph/Passes/RayTracingPass.h"

RayTracingPass::RayTracingPass(const Desc& desc)
    : RenderPass("RayTracing"), mDesc(desc) {}

void RayTracingPass::Setup(RenderGraph& graph, PassHandle self) {
    graph.Read(self, mDesc.depthResource, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    graph.Write(self, mDesc.colorResource, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
    graph.DependsOn(self, mDesc.depthResource, mDesc.forwardPassHandle);
    graph.DependsOn(self, mDesc.colorResource, mDesc.forwardPassHandle);
}

void RayTracingPass::Execute(VkCommandBuffer cmd) {
    auto tlas = mDesc.accel->GetTLAS();
    if (tlas == VK_NULL_HANDLE) return;

    if (mDesc.shadows && mDesc.shadows->IsEnabled()) {
        mDesc.shadows->Dispatch(cmd, tlas,
            mDesc.depthView, mDesc.depthSampler,
            mDesc.invViewProj,
            mDesc.lightDir, mDesc.lightRadius,
            mDesc.cameraPos);

        mDesc.shadows->Denoise(cmd, mDesc.depthView, mDesc.depthSampler, mDesc.invViewProj);
    }

    if (mDesc.reflections && mDesc.reflections->IsEnabled()) {
        mDesc.reflections->Dispatch(cmd, tlas,
            mDesc.depthView, mDesc.depthSampler,
            mDesc.invViewProj,
            mDesc.cameraPos, mDesc.roughness);

        mDesc.reflections->Denoise(cmd, mDesc.depthView, mDesc.depthSampler, mDesc.invViewProj);
    }

    // Composite RT results into HDR
    if (mDesc.compositePipeline != VK_NULL_HANDLE && mDesc.compositeDescSet != VK_NULL_HANDLE) {
        struct CompositePushConstants {
            glm::uvec2 resolution;
            float shadowStrength;
            float reflectionStrength;
            uint32_t enableShadows;
            uint32_t enableReflections;
            uint32_t debugShadowVis;
        } pc{};
        pc.resolution          = {mDesc.extent.width, mDesc.extent.height};
        pc.shadowStrength      = mDesc.shadowStrength;
        pc.reflectionStrength  = mDesc.reflectionStrength;
        pc.enableShadows       = (mDesc.shadows && mDesc.shadows->IsEnabled()) ? 1u : 0u;
        pc.enableReflections   = (mDesc.reflections && mDesc.reflections->IsEnabled()) ? 1u : 0u;
        pc.debugShadowVis      = mDesc.debugShadowVis ? 1u : 0u;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mDesc.compositePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mDesc.compositePipeLayout,
                                0, 1, &mDesc.compositeDescSet, 0, nullptr);
        vkCmdPushConstants(cmd, mDesc.compositePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (mDesc.extent.width + 7) / 8, (mDesc.extent.height + 7) / 8, 1);
    }
}
