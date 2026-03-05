#include "RenderGraph/Passes/HiZBuildPass.h"
#include "GPU/HiZBuffer.h"

HiZBuildPass::HiZBuildPass(const Desc& desc)
    : RenderPass("HiZBuild"), mDesc(desc) {}

void HiZBuildPass::Setup(RenderGraph& graph, PassHandle self) {
    graph.Read(self, mDesc.depthResource, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    graph.DependsOn(self, mDesc.depthResource, mDesc.occluderDepthPassHandle);
}

void HiZBuildPass::Execute(VkCommandBuffer cmd) {
    mDesc.hiZ->BuildMipChain(cmd);
}
