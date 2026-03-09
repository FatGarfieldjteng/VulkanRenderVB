#include "RenderGraph/Passes/PostProcessPass.h"
#include "PostProcess/PostProcessStack.h"
#include "PostProcess/AutoExposure.h"
#include "PostProcess/SSAO.h"
#include "PostProcess/Bloom.h"
#include "PostProcess/ToneMapping.h"
#include "PostProcess/ColorGrading.h"
#include "RHI/VulkanUtils.h"

PostProcessPass::PostProcessPass(const Desc& desc)
    : RenderPass("PostProcess"), mDesc(desc) {}

void PostProcessPass::Setup(RenderGraph& graph, PassHandle self) {
    graph.Read(self, mDesc.hdrResource, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
               VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    graph.Read(self, mDesc.depthResource, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    graph.Write(self, mDesc.swapchainResource, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    graph.DependsOn(self, mDesc.hdrResource, mDesc.forwardPassHandle);
    graph.DependsOn(self, mDesc.depthResource, mDesc.forwardPassHandle);
}

void PostProcessPass::Execute(VkCommandBuffer cmd) {
    mDesc.stack->TransitionPlaceholders(cmd);

    auto& settings = mDesc.stack->GetSettings();
    uint32_t w = mDesc.extent.width;
    uint32_t h = mDesc.extent.height;

    auto* exposure = mDesc.stack->GetAutoExposure();
    auto* ssao     = mDesc.stack->GetSSAO();
    auto* bloom    = mDesc.stack->GetBloom();
    auto* tonemap  = mDesc.stack->GetToneMapping();
    auto* cg       = mDesc.stack->GetColorGrading();

    VkImageView hdrView = mDesc.stack->GetHDRView();

    if (settings.autoExposureEnabled && exposure) {
        exposure->Dispatch(cmd, hdrView, VK_NULL_HANDLE, w, h,
                           settings.exposureMinEV, settings.exposureMaxEV,
                           mDesc.deltaTime, settings.adaptSpeed);
    }

    if (settings.ssaoEnabled && ssao) {
        ssao->Dispatch(cmd, mDesc.depthView,
                       mDesc.invProjection, mDesc.projInfo,
                       settings.ssaoRadius, settings.ssaoBias, settings.ssaoIntensity,
                       mDesc.farPlane, w, h);
    }

    if (settings.bloomEnabled && bloom) {
        bloom->Dispatch(cmd, hdrView, VK_NULL_HANDLE, w, h);
    }

    VkImageView aoView = (settings.ssaoEnabled && ssao) ? ssao->GetAOView()
                                                         : mDesc.stack->GetWhitePlaceholderView();
    VkImageView bloomView = (settings.bloomEnabled && bloom) ? bloom->GetBloomView()
                                                             : mDesc.stack->GetBlackPlaceholderView();
    VkBuffer exposureBuf = exposure ? exposure->GetExposureBuffer() : VK_NULL_HANDLE;

    ToneMappingPushConstants pc{};
    pc.curveType        = static_cast<uint32_t>(settings.toneCurve);
    pc.exposureBias     = settings.exposureBias;
    pc.whitePoint       = settings.acesWhitePoint;
    pc.shoulderStrength = settings.acesShoulderStrength;
    pc.linearStrength   = settings.acesLinearStrength;
    pc.linearAngle      = settings.acesLinearAngle;
    pc.toeStrength      = settings.acesToeStrength;
    pc.saturation       = settings.agxSaturation;
    pc.agxPunch         = settings.agxPunch;
    pc.bloomStrength    = settings.bloomEnabled ? settings.bloomStrength : 0.0f;
    pc.useAutoExposure  = settings.autoExposureEnabled ? 1u : 0u;

    bool doColorGrading = settings.colorGradingEnabled && cg;

    if (tonemap) {
        if (doColorGrading) {
            // Tone map → LDR intermediate, then color grade → swapchain
            VkImageView ldrView = mDesc.stack->GetLDRView();
            VkImage ldrImage    = mDesc.stack->GetLDRImage();

            TransitionImage(cmd, ldrImage,
                            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

            tonemap->Draw(cmd, ldrView, mDesc.extent,
                          hdrView, bloomView, aoView, exposureBuf, pc);

            TransitionImage(cmd, ldrImage,
                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            ColorGradingPushConstants cgPC{};
            cgPC.lutStrength         = settings.lutStrength;
            cgPC.vignetteIntensity   = settings.vignetteIntensity;
            cgPC.vignetteRadius      = settings.vignetteRadius;
            cgPC.grainStrength       = settings.grainStrength;
            cgPC.grainTime           = mDesc.deltaTime * 1000.0f;
            cgPC.chromaticAberration = settings.chromaticAberration;
            cgPC.resolutionX         = w;
            cgPC.resolutionY         = h;

            cg->Draw(cmd, mDesc.swapchainView, mDesc.extent, ldrView, cgPC);
        } else {
            tonemap->Draw(cmd, mDesc.swapchainView, mDesc.extent,
                          hdrView, bloomView, aoView, exposureBuf, pc);
        }
    }
}
