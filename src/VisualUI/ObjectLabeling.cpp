#include "VisualUI/ObjectLabeling.h"

namespace ObjectLabeling {

void SetName(VkDevice device, VkObjectType type, uint64_t handle, const char* name) {
    if (!vkSetDebugUtilsObjectNameEXT) return;

    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType   = type;
    info.objectHandle = handle;
    info.pObjectName  = name;
    vkSetDebugUtilsObjectNameEXT(device, &info);
}

void BeginLabel(VkCommandBuffer cmd, const char* name, float r, float g, float b, float a) {
    if (!vkCmdBeginDebugUtilsLabelEXT) return;

    VkDebugUtilsLabelEXT label{};
    label.sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    label.color[0]   = r;
    label.color[1]   = g;
    label.color[2]   = b;
    label.color[3]   = a;
    vkCmdBeginDebugUtilsLabelEXT(cmd, &label);
}

void EndLabel(VkCommandBuffer cmd) {
    if (!vkCmdEndDebugUtilsLabelEXT) return;
    vkCmdEndDebugUtilsLabelEXT(cmd);
}

} // namespace ObjectLabeling
