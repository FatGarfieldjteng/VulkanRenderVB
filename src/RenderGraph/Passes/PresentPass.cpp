#include "RenderGraph/Passes/PresentPass.h"

PresentPass::PresentPass(const Desc& desc)
    : RenderPass("Present"), mDesc(desc) {}

void PresentPass::Setup(RenderGraph& graph, PassHandle self) {
    graph.Write(self, mDesc.swapchainResource, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
    graph.DependsOn(self, mDesc.swapchainResource, mDesc.forwardPassHandle);
}

void PresentPass::Execute(VkCommandBuffer) {
}
