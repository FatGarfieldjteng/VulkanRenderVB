# DescriptorManager Design

## The Problem It Solves

In traditional Vulkan rendering, each material or texture needs its own descriptor set. If you have 500 textures, you'd need 500 descriptor sets and would constantly call `vkCmdBindDescriptorSets` between draw calls. This is expensive.

**Bindless descriptors** solve this by putting *all* textures into a single giant array inside a single descriptor set. The shader indexes into the array using a per-draw push constant. You bind the descriptor set once and never switch it.

## The Concept: One Giant Array

Imagine a large array of 16,384 slots:

```
Descriptor Set (binding 0):
  [0] = DamagedHelmet base color texture
  [1] = DamagedHelmet normal map
  [2] = terrain diffuse texture
  [3] = (empty, not yet assigned)
  [4] = (empty)
  ...
  [16383] = (empty)
```

When drawing an object, the shader receives the index via push constants and samples from `textures[index]`.

## Member Variables

```cpp
class DescriptorManager {
public:
    static constexpr uint32_t MAX_TEXTURES = 16384;
    // ...
private:
    VkDescriptorSetLayout mLayout         = VK_NULL_HANDLE;  // describes the array shape
    VkDescriptorPool      mPool           = VK_NULL_HANDLE;  // memory pool for descriptors
    VkDescriptorSet       mSet            = VK_NULL_HANDLE;  // THE single descriptor set
    VkSampler             mDefaultSampler = VK_NULL_HANDLE;  // shared trilinear sampler
    std::vector<bool>     mUsed;                             // which slots are occupied
    uint32_t              mNextFree = 0;                     // search hint for allocation
};
```

- **`mLayout`** -- Tells Vulkan "binding 0 is an array of 16384 combined image samplers"
- **`mPool`** -- Pre-allocates GPU memory for all 16384 descriptors
- **`mSet`** -- The single descriptor set that is bound for all rendering
- **`mDefaultSampler`** -- A reusable sampler (linear filtering, repeat wrapping, 16x anisotropy)
- **`mUsed`** -- CPU-side bookkeeping: tracks which array slots have a texture assigned
- **`mNextFree`** -- Optimization hint so allocation doesn't always scan from index 0

---

## `Initialize` -- Setting up the Bindless System

The initialization creates 4 Vulkan objects in sequence:

### 1. Default Sampler

```cpp
VkSamplerCreateInfo samplerInfo{};
samplerInfo.magFilter               = VK_FILTER_LINEAR;
samplerInfo.minFilter               = VK_FILTER_LINEAR;
samplerInfo.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
samplerInfo.addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
samplerInfo.addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
samplerInfo.addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
samplerInfo.anisotropyEnable        = VK_TRUE;
samplerInfo.maxAnisotropy           = 16.0f;
samplerInfo.maxLod                  = VK_LOD_CLAMP_NONE;
VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &mDefaultSampler));
```

This creates one sampler shared by all textures. The settings mean:

- **Trilinear filtering**: linear mag + linear min + linear between mip levels (smoothest quality)
- **Repeat wrapping**: UVs outside [0,1] wrap around (tiling)
- **16x anisotropic filtering**: improves quality at oblique viewing angles
- **`VK_LOD_CLAMP_NONE`**: allows sampling any mip level (no artificial LOD clamping)

### 2. Descriptor Set Layout

```cpp
VkDescriptorSetLayoutBinding binding{};
binding.binding         = 0;
binding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
binding.descriptorCount = MAX_TEXTURES;
binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

VkDescriptorBindingFlags bindingFlags =
    VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
    VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
    VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;

// ...
layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
```

The layout says: "at binding 0, there is an array of up to 16384 combined image samplers, accessible from fragment shaders."

The three binding flags are critical for bindless to work:

| Flag | What it enables |
|------|----------------|
| `PARTIALLY_BOUND` | Not every slot in the array needs a valid texture. Slots that the shader never indexes into can be empty. Without this, all 16384 slots would need valid descriptors. |
| `UPDATE_AFTER_BIND` | You can call `vkUpdateDescriptorSets` to add/remove textures even after the set has been bound to a command buffer. This allows dynamic texture loading without recreating the set. |
| `VARIABLE_DESCRIPTOR_COUNT` | The actual number of descriptors can be specified at allocation time rather than being fixed at layout creation time. |

The layout-level flag `UPDATE_AFTER_BIND_POOL_BIT` tells Vulkan the pool supporting this layout allows post-bind updates.

### 3. Descriptor Pool

```cpp
VkDescriptorPoolSize poolSize{};
poolSize.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
poolSize.descriptorCount = MAX_TEXTURES;

VkDescriptorPoolCreateInfo poolInfo{};
poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
poolInfo.maxSets       = 1;
poolInfo.poolSizeCount = 1;
poolInfo.pPoolSizes    = &poolSize;
```

The pool pre-allocates space for 16384 combined image sampler descriptors in 1 set. The `UPDATE_AFTER_BIND_BIT` flag matches the layout.

### 4. Allocate the Single Set

```cpp
uint32_t variableCount = MAX_TEXTURES;
VkDescriptorSetVariableDescriptorCountAllocateInfo variableInfo{};
variableInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
variableInfo.descriptorSetCount = 1;
variableInfo.pDescriptorCounts  = &variableCount;

VkDescriptorSetAllocateInfo allocInfo{};
allocInfo.pNext              = &variableInfo;
allocInfo.descriptorPool     = mPool;
allocInfo.descriptorSetCount = 1;
allocInfo.pSetLayouts        = &mLayout;

VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &mSet));
```

This allocates the one descriptor set from the pool. The `variableInfo` specifies that this set actually uses 16384 descriptors (matching the maximum). This set lives for the entire application lifetime.

---

## `AllocateTextureIndex` / `FreeTextureIndex` -- Slot Management

```cpp
uint32_t DescriptorManager::AllocateTextureIndex() {
    for (uint32_t i = mNextFree; i < MAX_TEXTURES; i++) {
        if (!mUsed[i]) {
            mUsed[i]  = true;
            mNextFree = i + 1;
            return i;
        }
    }
    for (uint32_t i = 0; i < mNextFree && i < MAX_TEXTURES; i++) {
        if (!mUsed[i]) { ... }
    }
    LOG_ERROR("DescriptorManager: no free texture index");
    return UINT32_MAX;
}

void DescriptorManager::FreeTextureIndex(uint32_t index) {
    if (index < MAX_TEXTURES) {
        mUsed[index] = false;
        if (index < mNextFree) mNextFree = index;
    }
}
```

This is a simple **free-list allocator** using a boolean array:

- **Allocate**: Scans forward from `mNextFree` to find an unused slot. If it reaches the end, wraps around to scan from 0. Returns the index.
- **Free**: Marks the slot as unused. Updates `mNextFree` if the freed index is lower (so the next allocation finds it quickly).

These indices are purely CPU-side bookkeeping. They don't touch Vulkan at all -- they just tell the application "use slot 3 for your next texture."

---

## `UpdateTexture` -- Writing a Texture into a Slot

```cpp
void DescriptorManager::UpdateTexture(VkDevice device, uint32_t index,
                                      VkImageView view, VkSampler sampler)
{
    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler     = sampler;
    imageInfo.imageView   = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.dstSet          = mSet;
    write.dstBinding      = 0;
    write.dstArrayElement = index;   // <-- which slot in the array
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo      = &imageInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}
```

This writes a specific texture (image view + sampler) into slot `index` of the array. The key field is `dstArrayElement = index` -- it targets one element within the 16384-element array at binding 0.

Because the `UPDATE_AFTER_BIND` flag was set, this call is safe even if the descriptor set is currently bound in a command buffer that hasn't finished executing yet.

---

## How It All Connects (End-to-End Flow)

Here's the complete flow from texture loading to shader sampling:

**1. Application loads a texture and assigns it a slot:**

```cpp
uint32_t descIdx = mDescriptors.AllocateTextureIndex();     // e.g. returns 0
mDescriptors.UpdateTexture(device, descIdx, gpuTex.GetView(),
                           mDescriptors.GetDefaultSampler());
```

**2. Application creates a pipeline layout referencing the bindless set layout:**

```cpp
VkDescriptorSetLayout setLayouts[] = { mDescriptors.GetLayout() };
```

**3. During rendering, the set is bound once:**

```cpp
VkDescriptorSet bindlessSet = mDescriptors.GetSet();
vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        mPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
```

**4. For each mesh, a push constant carries the texture index:**

```cpp
MeshPushConstants pc{};
pc.textureIndex = mesh.textureIndex;  // e.g. 0
vkCmdPushConstants(cmd, mPipelineLayout, ..., &pc);
vkCmdDrawIndexed(cmd, ...);
```

**5. In the fragment shader, the index selects from the array:**

```glsl
layout(set = 0, binding = 0) uniform sampler2D textures[];

vec4 texColor = texture(textures[nonuniformEXT(pc.textureIndex)], fragTexCoord);
```

`nonuniformEXT` tells the GPU that `pc.textureIndex` might differ across invocations in the same wave/warp (e.g., if different triangles in the same draw call had different texture indices). This ensures correct behavior on all GPU architectures.

---

## Summary Diagram

```
CPU side:                                          GPU side (descriptor set):

AllocateTextureIndex() -> 0                        binding 0:
AllocateTextureIndex() -> 1                          [0] = helmet_basecolor + sampler
AllocateTextureIndex() -> 2                          [1] = helmet_normal + sampler
                                                     [2] = terrain_diffuse + sampler
UpdateTexture(dev, 0, helmetView, sampler)            [3] = (empty, PARTIALLY_BOUND)
UpdateTexture(dev, 1, normalView, sampler)            ...
UpdateTexture(dev, 2, terrainView, sampler)           [16383] = (empty)

Draw call: pushConstant.textureIndex = 0  --->  textures[0]  --> samples helmet_basecolor
Draw call: pushConstant.textureIndex = 2  --->  textures[2]  --> samples terrain_diffuse
```
