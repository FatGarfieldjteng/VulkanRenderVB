#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <functional>

struct ImageKey {
    VkFormat             format      = VK_FORMAT_UNDEFINED;
    uint32_t             width       = 0;
    uint32_t             height      = 0;
    VkImageUsageFlags    usage       = 0;
    VkImageAspectFlags   aspect      = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t             arrayLayers = 1;
    uint32_t             mipLevels   = 1;
    VkSampleCountFlagBits samples   = VK_SAMPLE_COUNT_1_BIT;
    VkImageTiling        tiling      = VK_IMAGE_TILING_OPTIMAL;

    bool operator==(const ImageKey& o) const {
        return format == o.format && width == o.width && height == o.height &&
               usage == o.usage && aspect == o.aspect && arrayLayers == o.arrayLayers &&
               mipLevels == o.mipLevels && samples == o.samples && tiling == o.tiling;
    }
};

namespace std {
template <> struct hash<ImageKey> {
    size_t operator()(const ImageKey& k) const noexcept {
        size_t h = 0;
        auto combine = [&](size_t v) { h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2); };
        combine(std::hash<uint32_t>{}(static_cast<uint32_t>(k.format)));
        combine(std::hash<uint32_t>{}(k.width));
        combine(std::hash<uint32_t>{}(k.height));
        combine(std::hash<uint32_t>{}(static_cast<uint32_t>(k.usage)));
        combine(std::hash<uint32_t>{}(static_cast<uint32_t>(k.aspect)));
        combine(std::hash<uint32_t>{}(k.arrayLayers));
        combine(std::hash<uint32_t>{}(k.mipLevels));
        combine(std::hash<uint32_t>{}(static_cast<uint32_t>(k.samples)));
        combine(std::hash<uint32_t>{}(static_cast<uint32_t>(k.tiling)));
        return h;
    }
};
}

struct CachedImage {
    ImageKey      key;
    VkImage       image      = VK_NULL_HANDLE;
    VkImageView   view       = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint32_t      lastUsedFrame = 0;
    bool          inUse         = false;
};

class ImageCache {
public:
    void Initialize(VkDevice device, VmaAllocator allocator);
    void Shutdown();

    CachedImage* Acquire(const ImageKey& key, uint32_t currentFrame);
    void         Release(CachedImage* img);

    void EvictUnused(uint32_t currentFrame, uint32_t maxIdleFrames);

private:
    CachedImage* CreateImage(const ImageKey& key, uint32_t currentFrame);

    VkDevice     mDevice    = VK_NULL_HANDLE;
    VmaAllocator mAllocator = VK_NULL_HANDLE;
    std::mutex   mMutex;

    std::unordered_map<ImageKey, std::vector<CachedImage*>> mPool;
    std::vector<std::unique_ptr<CachedImage>>               mOwned;
};
