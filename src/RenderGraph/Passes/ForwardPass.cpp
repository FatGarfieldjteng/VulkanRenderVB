#include "RenderGraph/Passes/ForwardPass.h"
#include "RHI/VulkanUtils.h"
#include "GPU/MeshPool.h"

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
    graph.Write(self, mDesc.colorResource, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    graph.DependsOn(self, mDesc.csmResource, mDesc.shadowPassHandle);

    if (mDesc.gpuDriven) {
        PassHandle cullDep = mDesc.occlusionEnabled
                           ? mDesc.occlusionTestPassHandle
                           : mDesc.frustumCullPassHandle;
        graph.DependsOn(self, mDesc.depthResource, cullDep);
    }
}

void ForwardPass::Execute(VkCommandBuffer cmd) {
    const bool msaa = mDesc.msaaSamples != VK_SAMPLE_COUNT_1_BIT
                   && mDesc.msaaColorView != VK_NULL_HANDLE;

    if (msaa) {
        TransitionImage(cmd, mDesc.msaaColorImage,
                        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        TransitionImage(cmd, mDesc.msaaDepthImage,
                        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    VkRenderingAttachmentInfo colorAtt{};
    colorAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAtt.imageView   = msaa ? mDesc.msaaColorView : mDesc.colorView;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp     = msaa ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.clearValue.color = {{0.02f, 0.02f, 0.04f, 1.0f}};

    if (msaa) {
        colorAtt.resolveMode       = VK_RESOLVE_MODE_AVERAGE_BIT;
        colorAtt.resolveImageView  = mDesc.resolveColorView;
        colorAtt.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    VkRenderingAttachmentInfo depthAtt{};
    depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAtt.imageView   = msaa ? mDesc.msaaDepthView : mDesc.depthView;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAtt.storeOp     = msaa ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue.depthStencil = {1.0f, 0};

    if (msaa) {
        depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.resolveMode       = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
        depthAtt.resolveImageView  = mDesc.resolveDepthView;
        depthAtt.resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    } else {
        depthAtt.loadOp = (mDesc.gpuDriven && mDesc.occlusionEnabled)
                        ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    }

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

    VkBuffer vb[] = { mDesc.meshPool->GetVertexBuffer() };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vb, offsets);
    vkCmdBindIndexBuffer(cmd, mDesc.meshPool->GetIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);

    if (mDesc.gpuDriven && mDesc.indirectPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mDesc.indirectPipeline);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                mDesc.indirectPipelineLayout, 0, 1, &mDesc.bindlessSet, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                mDesc.indirectPipelineLayout, 1, 1, &mDesc.frameDescSet, 0, nullptr);

        const uint32_t stride = sizeof(VkDrawIndexedIndirectCommand);

        vkCmdDrawIndexedIndirectCount(cmd,
            mDesc.occluderBuffer, 0,
            mDesc.occluderCountBuffer, 0,
            mDesc.maxOccluderCount, stride);

        if (mDesc.occlusionEnabled) {
            vkCmdDrawIndexedIndirectCount(cmd,
                mDesc.visibleBuffer, 0,
                mDesc.visibleCountBuffer, 0,
                mDesc.maxVisibleCount, stride);
        }
    } else {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mDesc.pipeline);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                mDesc.pipelineLayout, 0, 1, &mDesc.bindlessSet, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                mDesc.pipelineLayout, 1, 1, &mDesc.frameDescSet, 0, nullptr);

        const auto& drawCmds = mDesc.meshPool->GetDrawCommands();

        mDesc.registry->ForEachRenderable([&](Entity, const TransformComponent& tc,
                                              const MeshComponent& mc, const MaterialComponent& matc) {
            if (mc.meshIndex < 0 || mc.meshIndex >= static_cast<int>(drawCmds.size())) return;
            int matIdx = std::clamp(matc.materialIndex, 0, static_cast<int>(mDesc.gpuMaterials->size()) - 1);

            PBRPushConstants pc{};
            pc.model         = tc.worldMatrix;
            pc.materialIndex = static_cast<uint32_t>(matIdx);

            vkCmdPushConstants(cmd, mDesc.pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, static_cast<uint32_t>(sizeof(glm::mat4) + sizeof(uint32_t)), &pc);

            const auto& poolCmd = drawCmds[mc.meshIndex];
            vkCmdDrawIndexed(cmd, poolCmd.indexCount, 1, poolCmd.firstIndex, poolCmd.vertexOffset, 0);
        });
    }
    vkCmdEndRendering(cmd);
}
