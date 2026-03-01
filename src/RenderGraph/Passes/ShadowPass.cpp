#include "RenderGraph/Passes/ShadowPass.h"
#include "Core/Application.h"  // GPUMesh definition

ShadowPass::ShadowPass(const Desc& desc)
    : RenderPass("Shadow"), mDesc(desc) {}

void ShadowPass::Setup(RenderGraph& graph, PassHandle self) {
    mSelf = self;
    graph.Write(self, mDesc.csmResource, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
}

void ShadowPass::Execute(VkCommandBuffer cmd) {
    constexpr uint32_t SD = CascadedShadowMap::SHADOW_DIM;
    constexpr uint32_t CC = CascadedShadowMap::CASCADE_COUNT;

    for (uint32_t cascade = 0; cascade < CC; cascade++) {
        VkRenderingAttachmentInfo depthAtt{};
        depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAtt.imageView   = mDesc.csm->GetLayerView(cascade);
        depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        depthAtt.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo ri{};
        ri.sType            = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea       = {{0, 0}, {SD, SD}};
        ri.layerCount       = 1;
        ri.pDepthAttachment = &depthAtt;
        vkCmdBeginRendering(cmd, &ri);

        VkViewport vp{0, 0, float(SD), float(SD), 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0, 0}, {SD, SD}};
        vkCmdSetScissor(cmd, 0, 1, &sc);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mDesc.pipeline);

        mDesc.registry->ForEachRenderable([&](Entity, const TransformComponent& tc,
                                              const MeshComponent& mc, const MaterialComponent&) {
            if (mc.meshIndex >= static_cast<int>(mDesc.gpuMeshes->size())) return;
            glm::mat4 mvp = mDesc.csm->GetViewProj(cascade) * tc.worldMatrix;
            vkCmdPushConstants(cmd, mDesc.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(glm::mat4), &mvp);
            const auto& mesh = (*mDesc.gpuMeshes)[mc.meshIndex];
            VkBuffer vb[] = { mesh.vertexBuffer.GetHandle() };
            VkDeviceSize off[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
            vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.GetHandle(), 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
        });
        vkCmdEndRendering(cmd);
    }
}
