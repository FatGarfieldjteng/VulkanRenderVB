#pragma once

#include <volk.h>
#include <string>

class PipelineManager {
public:
    void Initialize(VkDevice device);
    void Shutdown();

    /// Save the pipeline cache to disk for faster subsequent loads.
    void SaveCache(const std::string& path) const;

    /// Load a previously saved pipeline cache from disk.
    void LoadCache(const std::string& path);

    VkPipelineCache GetCache() const { return mCache; }

private:
    VkDevice        mDevice = VK_NULL_HANDLE;
    VkPipelineCache mCache  = VK_NULL_HANDLE;
};
