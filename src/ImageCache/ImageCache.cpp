#include "ImageCache/ImageCache.h"
#include "Core/Logger.h"

#include <algorithm>
#include <memory>

void ImageCache::Initialize(VkDevice device, VmaAllocator allocator) {
    mDevice    = device;
    mAllocator = allocator;
}

void ImageCache::Shutdown() {
    std::lock_guard lock(mMutex);
    for (auto& entry : mOwned) {
        if (entry->view  != VK_NULL_HANDLE) vkDestroyImageView(mDevice, entry->view, nullptr);
        if (entry->image != VK_NULL_HANDLE) {
            vmaDestroyImage(mAllocator, entry->image, entry->allocation);
        }
    }
    mPool.clear();
    mOwned.clear();
    LOG_INFO("ImageCache shutdown -- all images destroyed");
}

CachedImage* ImageCache::Acquire(const ImageKey& key, uint32_t currentFrame) {
    std::lock_guard lock(mMutex);

    auto it = mPool.find(key);
    if (it != mPool.end()) {
        for (auto* img : it->second) {
            if (!img->inUse) {
                img->inUse         = true;
                img->lastUsedFrame = currentFrame;
                return img;
            }
        }
    }

    return CreateImage(key, currentFrame);
}

void ImageCache::Release(CachedImage* img) {
    if (!img) return;
    std::lock_guard lock(mMutex);
    img->inUse = false;
}

void ImageCache::EvictUnused(uint32_t currentFrame, uint32_t maxIdleFrames) {
    std::lock_guard lock(mMutex);

    uint32_t evicted = 0;
    for (auto poolIt = mPool.begin(); poolIt != mPool.end(); ) {
        auto& vec = poolIt->second;
        vec.erase(std::remove_if(vec.begin(), vec.end(), [&](CachedImage* img) {
            if (img->inUse) return false;
            if (currentFrame - img->lastUsedFrame <= maxIdleFrames) return false;

            if (img->view  != VK_NULL_HANDLE) vkDestroyImageView(mDevice, img->view, nullptr);
            if (img->image != VK_NULL_HANDLE) vmaDestroyImage(mAllocator, img->image, img->allocation);
            img->view       = VK_NULL_HANDLE;
            img->image      = VK_NULL_HANDLE;
            img->allocation = VK_NULL_HANDLE;
            ++evicted;
            return true;
        }), vec.end());

        if (vec.empty())
            poolIt = mPool.erase(poolIt);
        else
            ++poolIt;
    }

    mOwned.erase(std::remove_if(mOwned.begin(), mOwned.end(), [](const auto& p) {
        return p->image == VK_NULL_HANDLE;
    }), mOwned.end());

    if (evicted > 0)
        LOG_INFO("ImageCache: evicted {} unused images", evicted);
}

CachedImage* ImageCache::CreateImage(const ImageKey& key, uint32_t currentFrame) {
    VkImageCreateInfo imgCI{};
    imgCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCI.imageType     = VK_IMAGE_TYPE_2D;
    imgCI.format        = key.format;
    imgCI.extent        = {key.width, key.height, 1};
    imgCI.mipLevels     = key.mipLevels;
    imgCI.arrayLayers   = key.arrayLayers;
    imgCI.samples       = key.samples;
    imgCI.tiling        = key.tiling;
    imgCI.usage         = key.usage;
    imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    auto entry = std::make_unique<CachedImage>();
    entry->key            = key;
    entry->lastUsedFrame  = currentFrame;
    entry->inUse          = true;

    VmaAllocationInfo allocInfo{};
    VkResult res = vmaCreateImage(mAllocator, &imgCI, &allocCI,
                                  &entry->image, &entry->allocation, &allocInfo);
    if (res != VK_SUCCESS) {
        LOG_ERROR("ImageCache: vmaCreateImage failed ({})", static_cast<int>(res));
        return nullptr;
    }

    VkImageViewCreateInfo viewCI{};
    viewCI.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image                           = entry->image;
    viewCI.viewType                        = (key.arrayLayers > 1) ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                                                    : VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format                          = key.format;
    viewCI.subresourceRange.aspectMask     = key.aspect;
    viewCI.subresourceRange.baseMipLevel   = 0;
    viewCI.subresourceRange.levelCount     = key.mipLevels;
    viewCI.subresourceRange.baseArrayLayer = 0;
    viewCI.subresourceRange.layerCount     = key.arrayLayers;
    vkCreateImageView(mDevice, &viewCI, nullptr, &entry->view);

    CachedImage* ptr = entry.get();
    mOwned.push_back(std::move(entry));
    mPool[key].push_back(ptr);

    return ptr;
}
