#include "RenderGraph/Passes/ForwardPass.h"
#include "Core/Application.h"

#include <algorithm>

struct PBRPushConstants {
    glm::mat4 model;
    uint32_t  materialIndex;
};

ForwardPass::ForwardPass(const Desc& desc)
    : RenderPass("Forward"), mDesc(desc) {}

void ForwardPass::Setup(RenderGraph& graph, PassHandle self) {
    graph.Read(self, mDesc.csmResource, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    graph.Write(self, mDesc.depthResource, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    graph.Write(self, mDesc.swapchainResource, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    graph.DependsOn(self, mDesc.csmResource, mDesc.shadowPassHandle);
}

void ForwardPass::Execute(VkCommandBuffer cmd) {
    VkRenderingAttachmentInfo colorAtt{};
    colorAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAtt.imageView   = mDesc.swapchainView;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.clearValue.color = {{0.02f, 0.02f, 0.04f, 1.0f}};

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
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &colorAtt;
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

    mDesc.registry->ForEachRenderable([&](Entity, const TransformComponent& tc,
                                          const MeshComponent& mc, const MaterialComponent& matc) {
        if (mc.meshIndex >= static_cast<int>(mDesc.gpuMeshes->size())) return;
        int matIdx = std::clamp(matc.materialIndex, 0, static_cast<int>(mDesc.gpuMaterials->size()) - 1);

        PBRPushConstants pc{};
        pc.model         = tc.worldMatrix;
        pc.materialIndex = static_cast<uint32_t>(matIdx);

        vkCmdPushConstants(cmd, mDesc.pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, static_cast<uint32_t>(sizeof(glm::mat4) + sizeof(uint32_t)), &pc);

        const auto& mesh = (*mDesc.gpuMeshes)[mc.meshIndex];
        VkBuffer vb[] = { mesh.vertexBuffer.GetHandle() };
        VkDeviceSize off[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
        vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.GetHandle(), 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
    });
    vkCmdEndRendering(cmd);
}
