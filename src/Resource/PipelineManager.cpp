#include "Resource/PipelineManager.h"
#include "Core/Logger.h"

#include <fstream>
#include <vector>

void PipelineManager::Initialize(VkDevice device) {
    mDevice = device;

    VkPipelineCacheCreateInfo cacheInfo{};
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VK_CHECK(vkCreatePipelineCache(device, &cacheInfo, nullptr, &mCache));

    LOG_INFO("PipelineManager initialized");
}

void PipelineManager::Shutdown() {
    if (mCache != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(mDevice, mCache, nullptr);
        mCache = VK_NULL_HANDLE;
    }
    LOG_INFO("PipelineManager destroyed");
}

void PipelineManager::SaveCache(const std::string& path) const {
    size_t dataSize = 0;
    VK_CHECK(vkGetPipelineCacheData(mDevice, mCache, &dataSize, nullptr));
    std::vector<char> data(dataSize);
    VK_CHECK(vkGetPipelineCacheData(mDevice, mCache, &dataSize, data.data()));

    std::ofstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.write(data.data(), static_cast<std::streamsize>(dataSize));
        LOG_INFO("Pipeline cache saved ({} bytes) to {}", dataSize, path);
    }
}

void PipelineManager::LoadCache(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        LOG_INFO("No existing pipeline cache at {}", path);
        return;
    }

    auto fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> data(fileSize);
    file.seekg(0);
    file.read(data.data(), static_cast<std::streamsize>(fileSize));

    if (mCache != VK_NULL_HANDLE)
        vkDestroyPipelineCache(mDevice, mCache, nullptr);

    VkPipelineCacheCreateInfo cacheInfo{};
    cacheInfo.sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheInfo.initialDataSize = data.size();
    cacheInfo.pInitialData    = data.data();
    VK_CHECK(vkCreatePipelineCache(mDevice, &cacheInfo, nullptr, &mCache));

    LOG_INFO("Pipeline cache loaded ({} bytes) from {}", fileSize, path);
}
