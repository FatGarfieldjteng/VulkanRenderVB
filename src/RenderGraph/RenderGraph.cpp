#include "RenderGraph/RenderGraph.h"
#include "RenderGraph/RenderPass.h"
#include "ImageCache/ImageCache.h"
#include "Core/Logger.h"

#include <algorithm>
#include <queue>

// =======================================================================
// Init / Shutdown
// =======================================================================

void RenderGraph::Initialize(VkDevice device, ImageCache* imageCache) {
    mDevice     = device;
    mImageCache = imageCache;
}

void RenderGraph::Shutdown() {
    ReleaseTransientResources();
}

// =======================================================================
// 1. Resources
// =======================================================================

RenderGraph::ResourceHandle RenderGraph::AddImage(
    const std::string& name, VkImage image, VkImageView view,
    VkImageLayout currentLayout, VkImageAspectFlags aspect, uint32_t arrayLayers)
{
    ResourceHandle h = static_cast<ResourceHandle>(mResources.size());
    ResourceNode node{};
    node.name          = name;
    node.image         = image;
    node.view          = view;
    node.initialLayout = currentLayout;
    node.aspect        = aspect;
    node.arrayLayers   = arrayLayers;
    node.isTransient   = false;
    mResources.push_back(std::move(node));
    return h;
}

RenderGraph::ResourceHandle RenderGraph::CreateImage(
    const std::string& name, const TransientImageDesc& desc)
{
    ResourceHandle h = static_cast<ResourceHandle>(mResources.size());
    ResourceNode node{};
    node.name          = name;
    node.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    node.aspect        = desc.aspect;
    node.arrayLayers   = desc.arrayLayers;
    node.isTransient   = true;
    node.transientDesc = desc;
    mResources.push_back(std::move(node));
    return h;
}

// =======================================================================
// 2. Passes
// =======================================================================

RenderGraph::PassHandle RenderGraph::AddPass(std::unique_ptr<RenderPass> pass) {
    PassHandle h = static_cast<PassHandle>(mPasses.size());
    mPasses.push_back({std::move(pass), {}, {}, {}});
    mPasses[h].pass->Setup(*this, h);
    return h;
}

// =======================================================================
// 3. Resource access
// =======================================================================

void RenderGraph::Read(PassHandle pass, ResourceHandle res, VkImageLayout layout,
                       VkPipelineStageFlags2 stage, VkAccessFlags2 access) {
    mPasses[pass].reads.push_back({res, layout, stage, access});
}

void RenderGraph::Write(PassHandle pass, ResourceHandle res, VkImageLayout layout,
                        VkPipelineStageFlags2 stage, VkAccessFlags2 access) {
    mPasses[pass].writes.push_back({res, layout, stage, access});
}

// =======================================================================
// 4. Explicit dependency
// =======================================================================

void RenderGraph::DependsOn(PassHandle pass, ResourceHandle resource, PassHandle dependency) {
    mPasses[pass].dependencies.push_back({dependency, resource});
}

// =======================================================================
// 5. Compile
// =======================================================================

void RenderGraph::Compile() {
    uint32_t passCount = static_cast<uint32_t>(mPasses.size());

    std::vector<std::vector<uint32_t>> adj(passCount);
    std::vector<uint32_t> inDegree(passCount, 0);

    for (uint32_t i = 0; i < passCount; i++) {
        for (const auto& dep : mPasses[i].dependencies) {
            adj[dep.dependency].push_back(i);
            inDegree[i]++;
        }
    }

    mExecutionOrder.clear();
    std::queue<uint32_t> ready;
    for (uint32_t i = 0; i < passCount; i++) {
        if (inDegree[i] == 0) ready.push(i);
    }
    while (!ready.empty()) {
        uint32_t p = ready.front();
        ready.pop();
        mExecutionOrder.push_back(p);
        for (uint32_t next : adj[p]) {
            if (--inDegree[next] == 0) ready.push(next);
        }
    }

    if (mExecutionOrder.size() != passCount) {
        LOG_ERROR("RenderGraph: cycle detected — falling back to declaration order");
        mExecutionOrder.clear();
        for (uint32_t i = 0; i < passCount; i++) mExecutionOrder.push_back(i);
    }

    ComputeLifetimes();
    AllocateTransientResources();

    mCompiled = true;
    LOG_INFO("RenderGraph compiled: {} passes, {} resources ({} transient)",
             passCount, mResources.size(),
             std::count_if(mResources.begin(), mResources.end(),
                           [](const ResourceNode& n) { return n.isTransient; }));
    for (uint32_t idx : mExecutionOrder)
        LOG_INFO("  [{}] {}", idx, mPasses[idx].pass->GetName());
}

// =======================================================================
// Lifetime computation
// =======================================================================

void RenderGraph::ComputeLifetimes() {
    for (auto& res : mResources) {
        res.firstUse = UINT32_MAX;
        res.lastUse  = 0;
    }

    for (uint32_t order = 0; order < mExecutionOrder.size(); order++) {
        const auto& entry = mPasses[mExecutionOrder[order]];

        auto touch = [&](ResourceHandle h) {
            if (h >= mResources.size()) return;
            mResources[h].firstUse = std::min(mResources[h].firstUse, order);
            mResources[h].lastUse  = std::max(mResources[h].lastUse, order);
        };

        for (const auto& r : entry.reads)  touch(r.resource);
        for (const auto& w : entry.writes) touch(w.resource);
    }
}

// =======================================================================
// Transient resource allocation via ImageCache
// =======================================================================

void RenderGraph::AllocateTransientResources() {
    for (uint32_t i = 0; i < static_cast<uint32_t>(mResources.size()); i++) {
        auto& res = mResources[i];
        if (!res.isTransient) continue;

        const auto& desc = res.transientDesc;
        ImageKey key{};
        key.format      = desc.format;
        key.width       = desc.width;
        key.height      = desc.height;
        key.usage       = desc.usage;
        key.aspect      = desc.aspect;
        key.arrayLayers = desc.arrayLayers;

        CachedImage* cached = mImageCache->Acquire(key, mFrameNumber);
        if (cached) {
            res.image = cached->image;
            res.view  = cached->view;
            TransientRef ref;
            ref.resource = i;
            ref.cached   = cached;
            mTransientRefs.push_back(ref);
        }
    }
}

// =======================================================================
// Release transient resources back to cache
// =======================================================================

void RenderGraph::ReleaseTransientResources() {
    if (!mImageCache) return;
    for (auto& ref : mTransientRefs) {
        mImageCache->Release(ref.cached);
    }
    mTransientRefs.clear();
}

// =======================================================================
// Execute: barriers via BarrierBatcher + pass execution
// =======================================================================

void RenderGraph::Execute(VkCommandBuffer cmd) {
    if (!mCompiled) {
        LOG_ERROR("RenderGraph::Execute called before Compile");
        return;
    }

    uint32_t resCount = static_cast<uint32_t>(mResources.size());
    mBarrierBatcher.Reset(resCount);

    for (uint32_t i = 0; i < resCount; i++)
        mBarrierBatcher.SetInitialState(i, mResources[i].initialLayout);

    for (uint32_t passIdx : mExecutionOrder) {
        const auto& entry = mPasses[passIdx];

        for (const auto& r : entry.reads) {
            const auto& res = mResources[r.resource];
            mBarrierBatcher.TransitionImage(
                r.resource, res.image, res.aspect, res.arrayLayers,
                r.layout, r.stage, r.access);
        }
        for (const auto& w : entry.writes) {
            const auto& res = mResources[w.resource];
            mBarrierBatcher.TransitionImage(
                w.resource, res.image, res.aspect, res.arrayLayers,
                w.layout, w.stage, w.access);
        }

        mBarrierBatcher.Flush(cmd);
        entry.pass->Execute(cmd);
    }
}

// =======================================================================
// BeginFrame
// =======================================================================

void RenderGraph::BeginFrame(uint32_t frameNumber) {
    ReleaseTransientResources();
    mResources.clear();
    mPasses.clear();
    mExecutionOrder.clear();
    mCompiled    = false;
    mFrameNumber = frameNumber;
}
