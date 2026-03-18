#include "RenderGraph/Passes/HybridRTPass.h"

HybridRTPass::HybridRTPass(const Desc& desc)
    : RenderPass("HybridRT"), mDesc(desc) {}

void HybridRTPass::Setup(RenderGraph& graph, PassHandle self) {
    graph.Read(self, mDesc.depthResource, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    graph.Write(self, mDesc.colorResource, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
    graph.DependsOn(self, mDesc.depthResource, mDesc.forwardPassHandle);
    graph.DependsOn(self, mDesc.colorResource, mDesc.forwardPassHandle);
}

void HybridRTPass::Execute(VkCommandBuffer cmd) {
    if (!mDesc.pathTracer) return;

    mDesc.pathTracer->Trace(cmd,
        mDesc.invViewProj, mDesc.viewProj,
        mDesc.cameraPos, mDesc.sunDir,
        mDesc.sunColor, mDesc.sunIntensity, mDesc.lightRadius,
        mDesc.denoiser && mDesc.enableDenoiser);

    if (mDesc.denoiser && mDesc.enableDenoiser) {
        bool cameraMoved = mDesc.pathTracer->WasAccumulationReset();
        if (cameraMoved)
            mDesc.denoiser->InvalidateHistory();

        // Synchronize RT writes → Compute reads
        VkMemoryBarrier2 rtToCompute{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        rtToCompute.srcStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        rtToCompute.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        rtToCompute.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        rtToCompute.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo rtDep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        rtDep.memoryBarrierCount = 1; rtDep.pMemoryBarriers = &rtToCompute;
        vkCmdPipelineBarrier2(cmd, &rtDep);

        // Falcor/paper: SVGF expects raw 1-SPP input
        mDesc.denoiser->Denoise(cmd,
            mDesc.pathTracer->GetColorOutputView(),
            mDesc.pathTracer->GetNormalOutputView(),
            mDesc.pathTracer->GetDepthOutputView(),
            mDesc.pathTracer->GetMotionOutputView(),
            mDesc.pathTracer->GetAlbedoOutputView(),
            mDesc.pathTracer->GetDepthOutputImage(),
            mDesc.pathTracer->GetNormalOutputImage(),
            mDesc.invViewProj,
            mDesc.viewProj,
            mDesc.viewMat,
            mDesc.projMat,
            mDesc.viewMatPrev,
            mDesc.projMatPrev,
            cameraMoved);
    }

    // Composite the RT indirect lighting into the HDR image (additive blend)
    if (mDesc.compositePipeline != VK_NULL_HANDLE && mDesc.compositeDescSet != VK_NULL_HANDLE) {
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.memoryBarrierCount = 1; dep.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &dep);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mDesc.compositePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mDesc.compositePipeLayout,
                                0, 1, &mDesc.compositeDescSet, 0, nullptr);
        if (mDesc.compareSplitX > 0) {
            struct ComparePC {
                glm::uvec2 resolution;
                uint32_t   splitLeft;
                uint32_t   splitRight;
                uint32_t   compareSplitX;
            } pc;
            pc.resolution    = {mDesc.extent.width, mDesc.extent.height};
            pc.splitLeft     = 0;
            pc.splitRight    = mDesc.extent.width;
            pc.compareSplitX = mDesc.compareSplitX;
            vkCmdPushConstants(cmd, mDesc.compositePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        } else {
            struct CopyPC {
                glm::uvec2 resolution;
                uint32_t   splitLeft;
                uint32_t   splitRight;
            } pc;
            pc.resolution = {mDesc.extent.width, mDesc.extent.height};
            pc.splitLeft  = 0;
            pc.splitRight = mDesc.extent.width;
            vkCmdPushConstants(cmd, mDesc.compositePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        }
        vkCmdDispatch(cmd, (mDesc.extent.width + 7) / 8, (mDesc.extent.height + 7) / 8, 1);
    }
}
