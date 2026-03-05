#include "RenderGraph/Passes/OccluderDepthPass.h"
#include "GPU/ComputeCulling.h"
#include "GPU/MeshPool.h"

OccluderDepthPass::OccluderDepthPass(const Desc& desc)
    : RenderPass("OccluderDepth"), mDesc(desc) {}

void OccluderDepthPass::Setup(RenderGraph& graph, PassHandle self) {
    graph.Write(self, mDesc.depthResource, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    graph.DependsOn(self, mDesc.depthResource, mDesc.frustumCullPassHandle);
}

void OccluderDepthPass::Execute(VkCommandBuffer cmd) {
    VkRenderingAttachmentInfo depthAtt{};
    depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAtt.imageView   = mDesc.depthView;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea           = {{0, 0}, mDesc.extent};
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 0;
    ri.pDepthAttachment     = &depthAtt;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{0, 0, float(mDesc.extent.width), float(mDesc.extent.height), 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{{0, 0}, mDesc.extent};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mDesc.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            mDesc.pipelineLayout, 0, 1, &mDesc.bindlessSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            mDesc.pipelineLayout, 1, 1, &mDesc.frameDescSet, 0, nullptr);

    VkBuffer vb[] = { mDesc.meshPool->GetVertexBuffer() };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vb, offsets);
    vkCmdBindIndexBuffer(cmd, mDesc.meshPool->GetIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexedIndirectCount(cmd,
        mDesc.culling->GetOccluderIndirectBuffer(), 0,
        mDesc.culling->GetOccluderCountBuffer(), 0,
        mDesc.maxDrawCount, sizeof(VkDrawIndexedIndirectCommand));

    vkCmdEndRendering(cmd);
}
