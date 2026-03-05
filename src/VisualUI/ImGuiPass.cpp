#include "VisualUI/ImGuiPass.h"
#include "VisualUI/DebugUI.h"

ImGuiPass::ImGuiPass(const Desc& desc)
    : RenderPass("ImGui"), mDesc(desc) {}

void ImGuiPass::Setup(RenderGraph& graph, PassHandle self) {
    graph.Write(self, mDesc.swapchainResource, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    graph.DependsOn(self, mDesc.swapchainResource, mDesc.previousPassHandle);
}

void ImGuiPass::Execute(VkCommandBuffer cmd) {
    mDesc.debugUI->Render(cmd, mDesc.swapchainView, mDesc.extent);
}
