#pragma once

#include <volk.h>
#include <string>
#include <cstdint>

struct TransientImageDesc {
    VkFormat             format      = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t             width       = 0;
    uint32_t             height      = 0;
    VkImageUsageFlags    usage       = 0;
    VkImageAspectFlags   aspect      = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t             arrayLayers = 1;
};

struct ResourceNode {
    std::string name;

    VkImage            image         = VK_NULL_HANDLE;
    VkImageView        view          = VK_NULL_HANDLE;
    VkImageLayout      initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageAspectFlags aspect        = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t           arrayLayers   = 1;

    bool               isTransient   = false;
    TransientImageDesc transientDesc{};

    uint32_t firstUse = UINT32_MAX;
    uint32_t lastUse  = 0;
};
