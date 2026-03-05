#pragma once

#include <volk.h>

namespace ObjectLabeling {

void SetName(VkDevice device, VkObjectType type, uint64_t handle, const char* name);

inline void NameBuffer(VkDevice device, VkBuffer buf, const char* name) {
    SetName(device, VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(buf), name);
}

inline void NameImage(VkDevice device, VkImage img, const char* name) {
    SetName(device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(img), name);
}

inline void NameImageView(VkDevice device, VkImageView view, const char* name) {
    SetName(device, VK_OBJECT_TYPE_IMAGE_VIEW, reinterpret_cast<uint64_t>(view), name);
}

inline void NamePipeline(VkDevice device, VkPipeline pipeline, const char* name) {
    SetName(device, VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(pipeline), name);
}

inline void NameDescriptorSet(VkDevice device, VkDescriptorSet set, const char* name) {
    SetName(device, VK_OBJECT_TYPE_DESCRIPTOR_SET, reinterpret_cast<uint64_t>(set), name);
}

inline void NameCommandBuffer(VkDevice device, VkCommandBuffer cmd, const char* name) {
    SetName(device, VK_OBJECT_TYPE_COMMAND_BUFFER, reinterpret_cast<uint64_t>(cmd), name);
}

void BeginLabel(VkCommandBuffer cmd, const char* name, float r = 0.4f, float g = 0.65f, float b = 1.0f, float a = 1.0f);
void EndLabel(VkCommandBuffer cmd);

struct ScopedLabel {
    VkCommandBuffer cmd;
    ScopedLabel(VkCommandBuffer c, const char* name, float r = 0.4f, float g = 0.65f, float b = 1.0f)
        : cmd(c) { BeginLabel(c, name, r, g, b); }
    ~ScopedLabel() { EndLabel(cmd); }
};

} // namespace ObjectLabeling
