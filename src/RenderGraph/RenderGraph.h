#pragma once

#include "RenderGraph/ResourceNode.h"
#include "RenderGraph/BarrierBatcher.h"
#include "RenderGraph/RenderPass.h"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

class ImageCache;
struct CachedImage;

class RenderGraph {
public:
    using PassHandle     = uint32_t;
    using ResourceHandle = uint32_t;
    static constexpr PassHandle     INVALID_PASS     = UINT32_MAX;
    static constexpr ResourceHandle INVALID_RESOURCE = UINT32_MAX;

    void Initialize(VkDevice device, ImageCache* imageCache);
    void Shutdown();

    // ----- 1. Resources -----

    ResourceHandle AddImage(const std::string& name, VkImage image, VkImageView view,
                            VkImageLayout currentLayout,
                            VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                            uint32_t arrayLayers = 1);

    ResourceHandle CreateImage(const std::string& name, const TransientImageDesc& desc);

    // ----- 2. Passes -----

    PassHandle AddPass(std::unique_ptr<RenderPass> pass);

    // ----- 3. Resource access (for barrier insertion) -----

    void Read(PassHandle pass, ResourceHandle res, VkImageLayout layout,
              VkPipelineStageFlags2 stage, VkAccessFlags2 access);

    void Write(PassHandle pass, ResourceHandle res, VkImageLayout layout,
               VkPipelineStageFlags2 stage, VkAccessFlags2 access);

    // ----- 4. Explicit dependency -----

    void DependsOn(PassHandle pass, ResourceHandle resource, PassHandle dependency);

    // ----- 5. Compile & Execute -----

    void Compile();
    void Execute(VkCommandBuffer cmd);

    void BeginFrame(uint32_t frameNumber);

    const ResourceNode& GetResource(ResourceHandle h) const { return mResources[h]; }

private:
    void ComputeLifetimes();
    void AllocateTransientResources();
    void ReleaseTransientResources();

    struct ResourceAccess {
        ResourceHandle        resource;
        VkImageLayout         layout;
        VkPipelineStageFlags2 stage;
        VkAccessFlags2        access;
    };

    struct Dependency {
        PassHandle     dependency;
        ResourceHandle resource;
    };

    struct PassEntry {
        std::unique_ptr<RenderPass>          pass;
        std::vector<ResourceAccess>          reads;
        std::vector<ResourceAccess>          writes;
        std::vector<Dependency>              dependencies;
    };

    VkDevice                          mDevice     = VK_NULL_HANDLE;
    ImageCache*                       mImageCache = nullptr;
    uint32_t                          mFrameNumber = 0;
    std::vector<ResourceNode>         mResources;
    std::vector<PassEntry>            mPasses;
    std::vector<uint32_t>             mExecutionOrder;
    BarrierBatcher                    mBarrierBatcher;
    bool                              mCompiled = false;

    struct TransientRef {
        ResourceHandle resource;
        CachedImage*   cached;
    };
    std::vector<TransientRef> mTransientRefs;
};
