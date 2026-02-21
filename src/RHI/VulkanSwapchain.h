#pragma once

#include <volk.h>
#include <vector>
#include <cstdint>

struct GLFWwindow;
struct QueueFamilyIndices;

class VulkanSwapchain {
public:
    static constexpr uint32_t DESIRED_IMAGE_COUNT = 3;

    void Initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                    VkSurfaceKHR surface, GLFWwindow* window,
                    const QueueFamilyIndices& indices);
    void Shutdown(VkDevice device);
    void Recreate(VkDevice device, VkPhysicalDevice physicalDevice,
                  VkSurfaceKHR surface, GLFWwindow* window,
                  const QueueFamilyIndices& indices);

    VkSwapchainKHR              GetHandle()      const { return mSwapchain; }
    VkFormat                    GetImageFormat()  const { return mImageFormat; }
    VkExtent2D                  GetExtent()       const { return mExtent; }
    const std::vector<VkImage>&     GetImages()     const { return mImages; }
    const std::vector<VkImageView>& GetImageViews() const { return mImageViews; }
    uint32_t                    GetImageCount()   const { return static_cast<uint32_t>(mImages.size()); }

private:
    void Create(VkDevice device, VkPhysicalDevice physicalDevice,
                VkSurfaceKHR surface, GLFWwindow* window,
                const QueueFamilyIndices& indices);
    void DestroyImageViews(VkDevice device);

    VkSurfaceFormatKHR ChooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const;
    VkPresentModeKHR   ChoosePresentMode(const std::vector<VkPresentModeKHR>& modes) const;
    VkExtent2D         ChooseExtent(const VkSurfaceCapabilitiesKHR& caps, GLFWwindow* window) const;

    VkSwapchainKHR           mSwapchain   = VK_NULL_HANDLE;
    VkFormat                 mImageFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D               mExtent      = {0, 0};
    std::vector<VkImage>     mImages;
    std::vector<VkImageView> mImageViews;
};
