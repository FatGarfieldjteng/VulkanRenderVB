#pragma once

#include <volk.h>
#include <string>
#include <cstdint>

class RenderGraph;

class RenderPass {
public:
    using PassHandle     = uint32_t;
    using ResourceHandle = uint32_t;

    explicit RenderPass(const std::string& name) : mName(name) {}
    virtual ~RenderPass() = default;

    const std::string& GetName() const { return mName; }

    virtual void Setup(RenderGraph& graph, PassHandle self) = 0;
    virtual void Execute(VkCommandBuffer cmd) = 0;

protected:
    std::string mName;
};
