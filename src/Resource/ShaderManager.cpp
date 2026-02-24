#include "Resource/ShaderManager.h"
#include "Core/Logger.h"

#include <fstream>

static std::vector<char> ReadBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open shader file: {}", path);
        return {};
    }
    auto fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    return buffer;
}

void ShaderManager::Initialize(VkDevice device) {
    mDevice = device;
    LOG_INFO("ShaderManager initialized");
}

void ShaderManager::Shutdown() {
    for (auto& [path, module] : mModules) {
        if (module != VK_NULL_HANDLE)
            vkDestroyShaderModule(mDevice, module, nullptr);
    }
    mModules.clear();
    LOG_INFO("ShaderManager destroyed");
}

VkShaderModule ShaderManager::GetOrLoad(const std::string& path) {
    auto it = mModules.find(path);
    if (it != mModules.end()) return it->second;

    auto code = ReadBinaryFile(path);
    if (code.empty()) return VK_NULL_HANDLE;

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode    = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(mDevice, &createInfo, nullptr, &module));

    mModules[path] = module;
    LOG_INFO("Shader loaded: {}", path);
    return module;
}
