#pragma once

#include <volk.h>
#include <string>
#include <unordered_map>
#include <vector>

class ShaderManager {
public:
    void Initialize(VkDevice device);
    void Shutdown();

    /// Load a SPIR-V shader from disk. Returns a cached module if already loaded.
    VkShaderModule GetOrLoad(const std::string& path);

private:
    VkDevice mDevice = VK_NULL_HANDLE;
    std::unordered_map<std::string, VkShaderModule> mModules;
};
