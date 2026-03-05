#include "RenderGraph/Passes/OcclusionTestPass.h"
#include "GPU/ComputeCulling.h"

OcclusionTestPass::OcclusionTestPass(const Desc& desc)
    : RenderPass("OcclusionTest"), mDesc(desc) {}

void OcclusionTestPass::Setup(RenderGraph& graph, PassHandle self) {
    graph.DependsOn(self, mDesc.depthResource, mDesc.hiZBuildPassHandle);
}

void OcclusionTestPass::Execute(VkCommandBuffer cmd) {
    VkMemoryBarrier2 hiZBarrier{};
    hiZBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    hiZBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    hiZBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    hiZBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    hiZBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo dep{};
    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &hiZBarrier;
    vkCmdPipelineBarrier2(cmd, &dep);

    mDesc.culling->DispatchOcclusion(cmd, *mDesc.params);
}
