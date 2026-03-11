#include "RayTracing/AccelStructure.h"
#include "Resource/TransferManager.h"
#include "Core/Logger.h"

#include <glm/gtc/type_ptr.hpp>
#include <cstring>
#include <numeric>

static constexpr VkDeviceSize kScratchAlign = 128;

static VkDeviceAddress AlignUp(VkDeviceAddress addr, VkDeviceSize alignment) {
    return (addr + alignment - 1) & ~(alignment - 1);
}

void AccelStructure::Initialize(VkDevice device, VmaAllocator allocator,
                                const TransferManager& transfer) {
    mDevice    = device;
    mAllocator = allocator;
    mTransfer  = &transfer;
}

void AccelStructure::Shutdown(VmaAllocator allocator) {
    if (mTLAS != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR(mDevice, mTLAS, nullptr);
        mTLAS = VK_NULL_HANDLE;
    }
    mTLASBuffer.Destroy(allocator);
    mInstanceBuffer.Destroy(allocator);

    for (auto& entry : mBLASEntries) {
        if (entry.handle != VK_NULL_HANDLE)
            vkDestroyAccelerationStructureKHR(mDevice, entry.handle, nullptr);
        entry.buffer.Destroy(allocator);
    }
    mBLASEntries.clear();
    mTotalBLASMemory = 0;
    mTotalBLASMemoryPreCompaction = 0;
    mTLASBuilt = false;
}

// ---------------------------------------------------------------------------
// BLAS — one per unique mesh in MeshPool
// ---------------------------------------------------------------------------
void AccelStructure::BuildBLAS(const MeshPool& meshPool) {
    const auto& cmds = meshPool.GetDrawCommands();
    if (cmds.empty()) return;

    const uint32_t meshCount = static_cast<uint32_t>(cmds.size());
    VkBuffer vertexBuf = meshPool.GetVertexBuffer();
    VkBuffer indexBuf  = meshPool.GetIndexBuffer();

    VkDeviceAddress vertexAddr{};
    {
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = vertexBuf;
        vertexAddr = vkGetBufferDeviceAddress(mDevice, &info);
    }
    VkDeviceAddress indexAddr{};
    {
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = indexBuf;
        indexAddr = vkGetBufferDeviceAddress(mDevice, &info);
    }

    std::vector<VkAccelerationStructureGeometryKHR>       geometries(meshCount);
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(meshCount);
    std::vector<VkAccelerationStructureBuildRangeInfoKHR>    rangeInfos(meshCount);
    std::vector<uint32_t> maxPrimCounts(meshCount);
    std::vector<VkAccelerationStructureBuildSizesInfoKHR>    sizeInfos(meshCount);

    for (uint32_t i = 0; i < meshCount; i++) {
        const auto& cmd = cmds[i];

        auto& geom = geometries[i];
        geom = {};
        geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;

        auto& tri = geom.geometry.triangles;
        tri.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tri.vertexFormat  = VK_FORMAT_R32G32B32_SFLOAT;
        tri.vertexData.deviceAddress = vertexAddr + cmd.vertexOffset * sizeof(MeshVertex);
        tri.vertexStride  = sizeof(MeshVertex);
        tri.maxVertex     = cmd.vertexCount > 0 ? cmd.vertexCount - 1 : 0;
        tri.indexType     = VK_INDEX_TYPE_UINT32;
        tri.indexData.deviceAddress = indexAddr + cmd.firstIndex * sizeof(uint32_t);

        maxPrimCounts[i] = cmd.indexCount / 3;

        auto& bi = buildInfos[i];
        bi = {};
        bi.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        bi.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bi.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                           VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
        bi.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bi.geometryCount = 1;
        bi.pGeometries   = &geometries[i];

        auto& range = rangeInfos[i];
        range = {};
        range.primitiveCount = maxPrimCounts[i];

        sizeInfos[i] = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        vkGetAccelerationStructureBuildSizesKHR(mDevice,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildInfos[i], &maxPrimCounts[i], &sizeInfos[i]);
    }

    // Allocate scratch buffer (max of all sizes)
    VkDeviceSize maxScratch = 0;
    for (uint32_t i = 0; i < meshCount; i++)
        maxScratch = std::max(maxScratch, sizeInfos[i].buildScratchSize);

    VulkanBuffer scratchBuffer;
    scratchBuffer.CreateDeviceLocalEmpty(mAllocator,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        maxScratch + kScratchAlign);
    VkDeviceAddress scratchAddr = AlignUp(scratchBuffer.GetDeviceAddress(mDevice), kScratchAlign);

    // Create query pool for compaction sizes
    VkQueryPool queryPool = VK_NULL_HANDLE;
    {
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType  = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
        qpci.queryCount = meshCount;
        VK_CHECK(vkCreateQueryPool(mDevice, &qpci, nullptr, &queryPool));
    }

    // Build all BLAS (one per mesh) and query compacted sizes
    mBLASEntries.resize(meshCount);
    mTotalBLASMemoryPreCompaction = 0;

    mTransfer->ImmediateSubmit([&](VkCommandBuffer cmd) {
        vkCmdResetQueryPool(cmd, queryPool, 0, meshCount);

        for (uint32_t i = 0; i < meshCount; i++) {
            auto& entry = mBLASEntries[i];
            VkDeviceSize asSize = sizeInfos[i].accelerationStructureSize;
            mTotalBLASMemoryPreCompaction += asSize;

            entry.buffer.CreateDeviceLocalEmpty(mAllocator,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, asSize);

            VkAccelerationStructureCreateInfoKHR createInfo{};
            createInfo.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            createInfo.buffer = entry.buffer.GetHandle();
            createInfo.size   = asSize;
            createInfo.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            VK_CHECK(vkCreateAccelerationStructureKHR(mDevice, &createInfo, nullptr, &entry.handle));

            buildInfos[i].dstAccelerationStructure  = entry.handle;
            buildInfos[i].scratchData.deviceAddress = scratchAddr;

            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfos[i];
            vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfos[i], &pRange);

            VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            barrier.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers    = &barrier;
            vkCmdPipelineBarrier2(cmd, &dep);

            vkCmdWriteAccelerationStructuresPropertiesKHR(cmd, 1, &entry.handle,
                VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, i);
        }
    });

    // Read compacted sizes
    std::vector<VkDeviceSize> compactedSizes(meshCount);
    VK_CHECK(vkGetQueryPoolResults(mDevice, queryPool, 0, meshCount,
        meshCount * sizeof(VkDeviceSize), compactedSizes.data(),
        sizeof(VkDeviceSize), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));

    for (uint32_t i = 0; i < meshCount; i++)
        mBLASEntries[i].compactedSize = compactedSizes[i];

    vkDestroyQueryPool(mDevice, queryPool, nullptr);
    scratchBuffer.Destroy(mAllocator);

    CompactBLAS();

    LOG_INFO("BLAS built: {} meshes, pre-compaction {:.1f} KB, post-compaction {:.1f} KB ({:.0f}% reduction)",
             meshCount,
             mTotalBLASMemoryPreCompaction / 1024.0f,
             mTotalBLASMemory / 1024.0f,
             mTotalBLASMemoryPreCompaction > 0
                 ? (1.0f - float(mTotalBLASMemory) / float(mTotalBLASMemoryPreCompaction)) * 100.0f
                 : 0.0f);
}

// ---------------------------------------------------------------------------
// BLAS compaction — copy each BLAS into a tighter allocation
// ---------------------------------------------------------------------------
void AccelStructure::CompactBLAS() {
    mTotalBLASMemory = 0;

    struct OldResource {
        VkAccelerationStructureKHR handle;
        VulkanBuffer               buffer;
    };
    std::vector<OldResource> oldResources;
    std::vector<VkAccelerationStructureKHR> newHandles(mBLASEntries.size(), VK_NULL_HANDLE);
    std::vector<VulkanBuffer>               newBuffers(mBLASEntries.size());

    mTransfer->ImmediateSubmit([&](VkCommandBuffer cmd) {
        for (size_t i = 0; i < mBLASEntries.size(); i++) {
            auto& entry = mBLASEntries[i];
            if (entry.compactedSize == 0 || entry.compactedSize >= entry.buffer.GetSize()) {
                mTotalBLASMemory += entry.buffer.GetSize();
                continue;
            }

            newBuffers[i].CreateDeviceLocalEmpty(mAllocator,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                entry.compactedSize);

            VkAccelerationStructureCreateInfoKHR createInfo{};
            createInfo.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            createInfo.buffer = newBuffers[i].GetHandle();
            createInfo.size   = entry.compactedSize;
            createInfo.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            VK_CHECK(vkCreateAccelerationStructureKHR(mDevice, &createInfo, nullptr, &newHandles[i]));

            VkCopyAccelerationStructureInfoKHR copyInfo{};
            copyInfo.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
            copyInfo.src   = entry.handle;
            copyInfo.dst   = newHandles[i];
            copyInfo.mode  = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
            vkCmdCopyAccelerationStructureKHR(cmd, &copyInfo);

            oldResources.push_back({entry.handle, entry.buffer});
            mTotalBLASMemory += entry.compactedSize;
        }
    });

    for (size_t i = 0; i < mBLASEntries.size(); i++) {
        if (newHandles[i] != VK_NULL_HANDLE) {
            mBLASEntries[i].handle = newHandles[i];
            mBLASEntries[i].buffer = newBuffers[i];
        }
    }

    for (auto& old : oldResources) {
        vkDestroyAccelerationStructureKHR(mDevice, old.handle, nullptr);
        old.buffer.Destroy(mAllocator);
    }
}

// ---------------------------------------------------------------------------
// TLAS — one instance per renderable entity
// ---------------------------------------------------------------------------
void AccelStructure::BuildTLAS(const Registry& registry, const MeshPool& meshPool) {
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    uint32_t instanceIdx = 0;

    registry.ForEachRenderable(
        [&](uint32_t, const TransformComponent& xform, const MeshComponent& mesh, const MaterialComponent&) {
            if (mesh.meshIndex >= static_cast<int>(mBLASEntries.size())) return;
            const auto& blas = mBLASEntries[mesh.meshIndex];
            if (blas.handle == VK_NULL_HANDLE) return;

            VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
            addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
            addrInfo.accelerationStructure = blas.handle;
            VkDeviceAddress blasAddr = vkGetAccelerationStructureDeviceAddressKHR(mDevice, &addrInfo);

            VkAccelerationStructureInstanceKHR inst{};
            const glm::mat4& m = xform.worldMatrix;
            VkTransformMatrixKHR xformMat{};
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 4; c++)
                    xformMat.matrix[r][c] = m[c][r]; // GLM is column-major, Vulkan is row-major

            inst.transform                              = xformMat;
            inst.instanceCustomIndex                    = instanceIdx;
            inst.mask                                   = 0xFF;
            inst.instanceShaderBindingTableRecordOffset  = 0;
            inst.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            inst.accelerationStructureReference          = blasAddr;
            instances.push_back(inst);
            instanceIdx++;
        });

    if (instances.empty()) return;

    VkDeviceSize instancesSize = instances.size() * sizeof(VkAccelerationStructureInstanceKHR);

    mInstanceBuffer.Destroy(mAllocator);
    mInstanceBuffer.CreateDeviceLocal(mAllocator, *mTransfer,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        instances.data(), instancesSize);

    VkDeviceAddress instanceAddr = mInstanceBuffer.GetDeviceAddress(mDevice);

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geom.geometry.instances.arrayOfPointers    = VK_FALSE;
    geom.geometry.instances.data.deviceAddress = instanceAddr;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &geom;

    uint32_t primCount = static_cast<uint32_t>(instances.size());

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(mDevice,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primCount, &sizeInfo);

    if (mTLAS != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR(mDevice, mTLAS, nullptr);
        mTLAS = VK_NULL_HANDLE;
    }
    mTLASBuffer.Destroy(mAllocator);

    mTLASBuffer.CreateDeviceLocalEmpty(mAllocator,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        sizeInfo.accelerationStructureSize);

    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = mTLASBuffer.GetHandle();
    createInfo.size   = sizeInfo.accelerationStructureSize;
    createInfo.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    VK_CHECK(vkCreateAccelerationStructureKHR(mDevice, &createInfo, nullptr, &mTLAS));

    VulkanBuffer scratchBuffer;
    scratchBuffer.CreateDeviceLocalEmpty(mAllocator,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        sizeInfo.buildScratchSize + kScratchAlign);

    buildInfo.dstAccelerationStructure  = mTLAS;
    buildInfo.scratchData.deviceAddress = AlignUp(scratchBuffer.GetDeviceAddress(mDevice), kScratchAlign);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;

    mTransfer->ImmediateSubmit([&](VkCommandBuffer cmd) {
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);
    });

    scratchBuffer.Destroy(mAllocator);
    mTLASBuilt = true;

    LOG_INFO("TLAS built: {} instances, {:.1f} KB", primCount, sizeInfo.accelerationStructureSize / 1024.0f);
}

// ---------------------------------------------------------------------------
// TLAS update (incremental rebuild for dynamic objects)
// ---------------------------------------------------------------------------
void AccelStructure::UpdateTLAS(const Registry& registry, const MeshPool& meshPool) {
    if (!mTLASBuilt) {
        BuildTLAS(registry, meshPool);
        return;
    }

    std::vector<VkAccelerationStructureInstanceKHR> instances;
    uint32_t instanceIdx = 0;

    registry.ForEachRenderable(
        [&](uint32_t, const TransformComponent& xform, const MeshComponent& mesh, const MaterialComponent&) {
            if (mesh.meshIndex >= static_cast<int>(mBLASEntries.size())) return;
            const auto& blas = mBLASEntries[mesh.meshIndex];
            if (blas.handle == VK_NULL_HANDLE) return;

            VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
            addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
            addrInfo.accelerationStructure = blas.handle;
            VkDeviceAddress blasAddr = vkGetAccelerationStructureDeviceAddressKHR(mDevice, &addrInfo);

            VkAccelerationStructureInstanceKHR inst{};
            const glm::mat4& m = xform.worldMatrix;
            VkTransformMatrixKHR xformMat{};
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 4; c++)
                    xformMat.matrix[r][c] = m[c][r];

            inst.transform                              = xformMat;
            inst.instanceCustomIndex                    = instanceIdx;
            inst.mask                                   = 0xFF;
            inst.instanceShaderBindingTableRecordOffset  = 0;
            inst.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            inst.accelerationStructureReference          = blasAddr;
            instances.push_back(inst);
            instanceIdx++;
        });

    if (instances.empty()) return;

    VkDeviceSize instancesSize = instances.size() * sizeof(VkAccelerationStructureInstanceKHR);

    mInstanceBuffer.Destroy(mAllocator);
    mInstanceBuffer.CreateDeviceLocal(mAllocator, *mTransfer,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        instances.data(), instancesSize);

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geom.geometry.instances.arrayOfPointers    = VK_FALSE;
    geom.geometry.instances.data.deviceAddress = mInstanceBuffer.GetDeviceAddress(mDevice);

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type                     = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags                    = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                                         VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.srcAccelerationStructure = mTLAS;
    buildInfo.dstAccelerationStructure = mTLAS;
    buildInfo.geometryCount            = 1;
    buildInfo.pGeometries              = &geom;

    uint32_t primCount = static_cast<uint32_t>(instances.size());
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(mDevice,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primCount, &sizeInfo);

    VulkanBuffer scratchBuffer;
    scratchBuffer.CreateDeviceLocalEmpty(mAllocator,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        sizeInfo.updateScratchSize + kScratchAlign);

    buildInfo.scratchData.deviceAddress = AlignUp(scratchBuffer.GetDeviceAddress(mDevice), kScratchAlign);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;

    mTransfer->ImmediateSubmit([&](VkCommandBuffer cmd) {
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);
    });

    scratchBuffer.Destroy(mAllocator);
}
