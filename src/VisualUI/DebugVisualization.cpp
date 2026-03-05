#include "VisualUI/DebugVisualization.h"
#include "Resource/ShaderManager.h"
#include "Resource/PipelineManager.h"
#include "Asset/ModelLoader.h"
#include "Core/Logger.h"

void DebugVisualization::Initialize(VkDevice device, ShaderManager& shaders,
                                     PipelineManager& pipelines,
                                     VkFormat colorFormat,
                                     VkDescriptorSetLayout bindlessLayout,
                                     VkDescriptorSetLayout frameLayout) {
    CreateDebugPipeline(device, shaders, pipelines, colorFormat, bindlessLayout, frameLayout);
    LOG_INFO("DebugVisualization initialized");
}

void DebugVisualization::Shutdown(VkDevice device) {
    if (mWireframePipeline)  vkDestroyPipeline(device, mWireframePipeline, nullptr);
    if (mDebugModePipeline)  vkDestroyPipeline(device, mDebugModePipeline, nullptr);
    if (mPipelineLayout)     vkDestroyPipelineLayout(device, mPipelineLayout, nullptr);
    mWireframePipeline = VK_NULL_HANDLE;
    mDebugModePipeline = VK_NULL_HANDLE;
    mPipelineLayout    = VK_NULL_HANDLE;
}

VkPipeline DebugVisualization::GetPipeline(DebugUIState::VisMode mode) const {
    if (mode == DebugUIState::VisMode::Wireframe)
        return mWireframePipeline;
    if (mode != DebugUIState::VisMode::None)
        return mDebugModePipeline;
    return VK_NULL_HANDLE;
}

bool DebugVisualization::RequiresSpecialRendering(DebugUIState::VisMode mode) const {
    return mode != DebugUIState::VisMode::None;
}

void DebugVisualization::CreateDebugPipeline(VkDevice device, ShaderManager& shaders,
                                              PipelineManager& pipelines, VkFormat colorFormat,
                                              VkDescriptorSetLayout bindlessLayout,
                                              VkDescriptorSetLayout frameLayout)
{
    VkShaderModule vertModule = shaders.GetOrLoad("shaders/pbr_indirect.vert.spv");
    VkShaderModule fragModule = shaders.GetOrLoad("shaders/debug_vis.frag.spv");
    if (!vertModule || !fragModule) {
        LOG_WARN("Debug visualization shaders not found, visualization disabled");
        return;
    }

    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding   = 0;
    bindingDesc.stride    = sizeof(MeshVertex);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrDescs[4]{};
    attrDescs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(MeshVertex, position)};
    attrDescs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(MeshVertex, normal)};
    attrDescs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(MeshVertex, texCoord)};
    attrDescs[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(MeshVertex, tangent)};

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount   = 1;
    vertexInput.pVertexBindingDescriptions      = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = 4;
    vertexInput.pVertexAttributeDescriptions    = attrDescs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates    = dynStates;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments    = &blendAtt;

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(uint32_t);

    VkDescriptorSetLayout setLayouts[] = { bindlessLayout, frameLayout };
    VkPipelineLayoutCreateInfo layoutCI{};
    layoutCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCI.setLayoutCount         = 2;
    layoutCI.pSetLayouts            = setLayouts;
    layoutCI.pushConstantRangeCount = 1;
    layoutCI.pPushConstantRanges    = &pushRange;
    vkCreatePipelineLayout(device, &layoutCI, nullptr, &mPipelineLayout);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr};

    VkPipelineRenderingCreateInfo renderInfo{};
    renderInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderInfo.colorAttachmentCount    = 1;
    renderInfo.pColorAttachmentFormats = &colorFormat;
    renderInfo.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode    = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth   = 1.0f;

    VkGraphicsPipelineCreateInfo ci{};
    ci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    ci.pNext               = &renderInfo;
    ci.stageCount          = 2;
    ci.pStages             = stages;
    ci.pVertexInputState   = &vertexInput;
    ci.pInputAssemblyState = &inputAssembly;
    ci.pViewportState      = &viewportState;
    ci.pRasterizationState = &rasterizer;
    ci.pMultisampleState   = &multisampling;
    ci.pDepthStencilState  = &depthStencil;
    ci.pColorBlendState    = &colorBlend;
    ci.pDynamicState       = &dynamicState;
    ci.layout              = mPipelineLayout;

    vkCreateGraphicsPipelines(device, pipelines.GetCache(), 1, &ci, nullptr, &mDebugModePipeline);

    rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
    vkCreateGraphicsPipelines(device, pipelines.GetCache(), 1, &ci, nullptr, &mWireframePipeline);
}
