#include "Lighting/CascadedShadowMap.h"
#include "Core/Logger.h"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>
#include <array>

void CascadedShadowMap::Initialize(VmaAllocator allocator, VkDevice device) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.format        = VK_FORMAT_D32_SFLOAT;
    imageInfo.extent        = { SHADOW_DIM, SHADOW_DIM, 1 };
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = CASCADE_COUNT;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocCI, &mImage, &mAllocation, nullptr));

    // Full array view for shader sampling
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image    = mImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format   = VK_FORMAT_D32_SFLOAT;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = CASCADE_COUNT;
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &mArrayView));

    // Per-layer views for rendering
    for (uint32_t i = 0; i < CASCADE_COUNT; i++) {
        VkImageViewCreateInfo layerViewInfo{};
        layerViewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        layerViewInfo.image    = mImage;
        layerViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        layerViewInfo.format   = VK_FORMAT_D32_SFLOAT;
        layerViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
        layerViewInfo.subresourceRange.baseMipLevel   = 0;
        layerViewInfo.subresourceRange.levelCount     = 1;
        layerViewInfo.subresourceRange.baseArrayLayer = i;
        layerViewInfo.subresourceRange.layerCount     = 1;
        VK_CHECK(vkCreateImageView(device, &layerViewInfo, nullptr, &mLayerViews[i]));
    }

    // Comparison sampler for PCF
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter     = VK_FILTER_LINEAR;
    samplerInfo.minFilter     = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.borderColor   = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    samplerInfo.compareEnable = VK_TRUE;
    samplerInfo.compareOp     = VK_COMPARE_OP_LESS_OR_EQUAL;
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &mSampler));

    LOG_INFO("CascadedShadowMap initialized: {}x{} x {} cascades", SHADOW_DIM, SHADOW_DIM, CASCADE_COUNT);
}

void CascadedShadowMap::Shutdown(VmaAllocator allocator, VkDevice device) {
    if (mSampler)   vkDestroySampler(device, mSampler, nullptr);
    if (mArrayView) vkDestroyImageView(device, mArrayView, nullptr);
    for (auto& v : mLayerViews)
        if (v) vkDestroyImageView(device, v, nullptr);
    if (mImage) vmaDestroyImage(allocator, mImage, mAllocation);

    mSampler    = VK_NULL_HANDLE;
    mArrayView  = VK_NULL_HANDLE;
    mLayerViews = {};
    mImage      = VK_NULL_HANDLE;
    mAllocation = VK_NULL_HANDLE;
}

void CascadedShadowMap::Update(const glm::mat4& cameraView, const glm::mat4& cameraProj,
                               float cameraNear, float cameraFar,
                               const glm::vec3& lightDir) {
    // Practical split scheme: C_i = lambda * n*(f/n)^(i/N) + (1-lambda) * (n + (f-n)*i/N)
    float splits[CASCADE_COUNT + 1];
    splits[0] = cameraNear;
    for (uint32_t i = 1; i <= CASCADE_COUNT; i++) {
        float p = static_cast<float>(i) / static_cast<float>(CASCADE_COUNT);
        float logSplit = cameraNear * std::pow(cameraFar / cameraNear, p);
        float uniSplit = cameraNear + (cameraFar - cameraNear) * p;
        splits[i] = LAMBDA * logSplit + (1.0f - LAMBDA) * uniSplit;
    }

    mSplitDepths = glm::vec4(splits[1], splits[2], splits[3], splits[4]);

    glm::mat4 invViewProj = glm::inverse(cameraProj * cameraView);

    for (uint32_t c = 0; c < CASCADE_COUNT; c++) {
        float nearSplit = splits[c];
        float farSplit  = splits[c + 1];

        // Compute the 8 corners of this cascade's sub-frustum in NDC, then to world space
        // NDC z range: Vulkan uses [0,1] depth, but glm::perspective with GLM_FORCE_DEPTH_ZERO_TO_ONE
        // We remap near/far planes to [0,1] NDC z
        float zNear = (nearSplit - cameraNear) / (cameraFar - cameraNear);
        float zFar  = (farSplit  - cameraNear) / (cameraFar - cameraNear);

        // Actually we need to reverse-engineer the NDC z from the projection matrix.
        // For a standard perspective matrix:  z_ndc = (f*z - f*n) / (z*(f-n))  mapped to [0,1]
        // But it's easier to just build a new projection for the sub-frustum.
        // Instead, use the inverse of the full view-proj and clip to this cascade's range.

        // Frustum corners in NDC: x,y in [-1,1], z in [0,1] for Vulkan
        std::array<glm::vec4, 8> corners = {{
            {-1,  1, 0, 1}, { 1,  1, 0, 1}, { 1, -1, 0, 1}, {-1, -1, 0, 1},
            {-1,  1, 1, 1}, { 1,  1, 1, 1}, { 1, -1, 1, 1}, {-1, -1, 1, 1},
        }};

        // Transform NDC corners to world space (full frustum)
        for (auto& corner : corners) {
            corner = invViewProj * corner;
            corner /= corner.w;
        }

        // Linearly interpolate between near and far to get sub-frustum
        // nearFrac/farFrac = fraction along the full frustum depth
        float range = cameraFar - cameraNear;
        float nearFrac = (nearSplit - cameraNear) / range;
        float farFrac  = (farSplit  - cameraNear) / range;

        std::array<glm::vec4, 8> subCorners;
        for (int i = 0; i < 4; i++) {
            glm::vec4 nearCorner = corners[i];
            glm::vec4 farCorner  = corners[i + 4];
            subCorners[i]     = glm::mix(nearCorner, farCorner, nearFrac);
            subCorners[i + 4] = glm::mix(nearCorner, farCorner, farFrac);
        }

        // Compute center
        glm::vec3 center(0.0f);
        for (auto& sc : subCorners)
            center += glm::vec3(sc);
        center /= 8.0f;

        // Bounding sphere radius for stable shadow edges
        float radius = 0.0f;
        for (auto& sc : subCorners)
            radius = std::max(radius, glm::length(glm::vec3(sc) - center));

        // Round to texel size to reduce shimmering
        float texelSize = (radius * 2.0f) / static_cast<float>(SHADOW_DIM);

        glm::vec3 lightDirN = glm::normalize(lightDir);
        glm::mat4 lightView = glm::lookAt(center - lightDirN * radius,
                                           center,
                                           glm::vec3(0, 1, 0));

        // Snap the light view position to texel boundaries
        glm::vec4 shadowOrigin = lightView * glm::vec4(0, 0, 0, 1);
        shadowOrigin.x = std::floor(shadowOrigin.x / texelSize) * texelSize;
        shadowOrigin.y = std::floor(shadowOrigin.y / texelSize) * texelSize;

        glm::mat4 lightProj = glm::ortho(-radius, radius, -radius, radius,
                                          0.0f, radius * 2.0f);
        lightProj[1][1] *= -1.0f; // Vulkan Y-flip

        mViewProj[c] = lightProj * lightView;
    }
}
