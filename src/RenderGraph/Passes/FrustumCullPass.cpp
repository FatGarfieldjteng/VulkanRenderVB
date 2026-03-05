#include "RenderGraph/Passes/FrustumCullPass.h"
#include "GPU/ComputeCulling.h"

FrustumCullPass::FrustumCullPass(const Desc& desc)
    : RenderPass("FrustumCull"), mDesc(desc) {}

void FrustumCullPass::Setup(RenderGraph&, PassHandle) {}

void FrustumCullPass::Execute(VkCommandBuffer cmd) {
    mDesc.culling->DispatchFrustum(cmd, *mDesc.params);
}
