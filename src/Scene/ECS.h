#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <functional>

using Entity = uint32_t;
constexpr Entity INVALID_ENTITY = UINT32_MAX;

template<typename T>
class ComponentPool {
public:
    T& Add(Entity e) {
        Ensure(e);
        mValid[e] = 1;
        mData[e] = T{};
        return mData[e];
    }

    void Remove(Entity e) {
        if (e < mValid.size()) mValid[e] = 0;
    }

    T*       Get(Entity e)       { return (e < mValid.size() && mValid[e]) ? &mData[e] : nullptr; }
    const T* Get(Entity e) const { return (e < mValid.size() && mValid[e]) ? &mData[e] : nullptr; }

    template<typename Fn> void ForEach(Fn&& fn) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(mData.size()); i++)
            if (mValid[i]) fn(static_cast<Entity>(i), mData[i]);
    }
    template<typename Fn> void ForEach(Fn&& fn) const {
        for (uint32_t i = 0; i < static_cast<uint32_t>(mData.size()); i++)
            if (mValid[i]) fn(static_cast<Entity>(i), mData[i]);
    }

private:
    void Ensure(Entity e) {
        if (e >= static_cast<uint32_t>(mData.size())) {
            mData.resize(e + 1);
            mValid.resize(e + 1, 0);
        }
    }
    std::vector<T>       mData;
    std::vector<uint8_t> mValid;
};

struct TransformComponent {
    glm::vec3 localPosition{0.0f};
    glm::quat localRotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 localScale{1.0f};
    Entity    parent = INVALID_ENTITY;
    bool      dirty  = true;
    glm::mat4 worldMatrix{1.0f};

    glm::mat4 GetLocalMatrix() const {
        glm::mat4 T = glm::translate(glm::mat4(1.0f), localPosition);
        glm::mat4 R = glm::mat4_cast(localRotation);
        glm::mat4 S = glm::scale(glm::mat4(1.0f), localScale);
        return T * R * S;
    }
};

struct MeshComponent     { int meshIndex     = -1; };
struct MaterialComponent { int materialIndex = -1; };

struct LightComponent {
    enum class Type { Directional };
    Type      type      = Type::Directional;
    glm::vec3 direction{glm::normalize(glm::vec3(-0.4f, -0.8f, -0.3f))};
    glm::vec3 color{1.0f, 0.95f, 0.85f};
    float     intensity = 3.5f;
};

class Registry {
public:
    Entity CreateEntity() {
        Entity e;
        if (!mFreeList.empty()) {
            e = mFreeList.back();
            mFreeList.pop_back();
        } else {
            e = mNextId++;
        }
        mAlive.insert(e);
        return e;
    }

    void DestroyEntity(Entity e) {
        if (mAlive.erase(e) == 0) return;

        auto* tc = mTransforms.Get(e);
        if (tc && tc->parent != INVALID_ENTITY) {
            auto it = mChildren.find(tc->parent);
            if (it != mChildren.end()) {
                auto& ch = it->second;
                ch.erase(std::remove(ch.begin(), ch.end(), e), ch.end());
            }
        }
        auto it = mChildren.find(e);
        if (it != mChildren.end()) {
            for (Entity child : it->second) {
                auto* ct = mTransforms.Get(child);
                if (ct) ct->parent = INVALID_ENTITY;
            }
            mChildren.erase(it);
        }

        mTransforms.Remove(e);
        mMeshes.Remove(e);
        mMaterials.Remove(e);
        mLights.Remove(e);
        mFreeList.push_back(e);
    }

    bool IsAlive(Entity e) const { return mAlive.count(e) > 0; }

    TransformComponent& AddTransform(Entity e)  { return mTransforms.Add(e); }
    MeshComponent&      AddMesh(Entity e)       { return mMeshes.Add(e); }
    MaterialComponent&  AddMaterial(Entity e)    { return mMaterials.Add(e); }
    LightComponent&     AddLight(Entity e)       { return mLights.Add(e); }

    TransformComponent* GetTransform(Entity e)       { return mTransforms.Get(e); }
    MeshComponent*      GetMesh(Entity e)            { return mMeshes.Get(e); }
    MaterialComponent*  GetMaterial(Entity e)        { return mMaterials.Get(e); }
    LightComponent*     GetLight(Entity e)           { return mLights.Get(e); }

    const TransformComponent* GetTransform(Entity e) const { return mTransforms.Get(e); }
    const MeshComponent*      GetMesh(Entity e)      const { return mMeshes.Get(e); }
    const MaterialComponent*  GetMaterial(Entity e)   const { return mMaterials.Get(e); }
    const LightComponent*     GetLight(Entity e)      const { return mLights.Get(e); }

    void SetParent(Entity child, Entity parent) {
        auto* ct = mTransforms.Get(child);
        if (!ct) return;
        if (ct->parent != INVALID_ENTITY) {
            auto it = mChildren.find(ct->parent);
            if (it != mChildren.end()) {
                auto& ch = it->second;
                ch.erase(std::remove(ch.begin(), ch.end(), child), ch.end());
            }
        }
        ct->parent = parent;
        ct->dirty = true;
        if (parent != INVALID_ENTITY)
            mChildren[parent].push_back(child);
    }

    void UpdateTransforms() {
        mTransforms.ForEach([this](Entity e, TransformComponent& tc) {
            if (tc.parent == INVALID_ENTITY) {
                tc.worldMatrix = tc.GetLocalMatrix();
                tc.dirty = false;
                PropagateChildren(e, tc.worldMatrix);
            }
        });
    }

    Entity FindSunLight() const {
        Entity sun = INVALID_ENTITY;
        mLights.ForEach([&](Entity e, const LightComponent& lc) {
            if (lc.type == LightComponent::Type::Directional)
                sun = e;
        });
        return sun;
    }

    template<typename Fn>
    void ForEachRenderable(Fn&& fn) const {
        mTransforms.ForEach([&](Entity e, const TransformComponent& tc) {
            const MeshComponent* mc = mMeshes.Get(e);
            const MaterialComponent* matc = mMaterials.Get(e);
            if (mc && matc && mc->meshIndex >= 0)
                fn(e, tc, *mc, *matc);
        });
    }

    uint32_t EntityCount() const { return static_cast<uint32_t>(mAlive.size()); }

private:
    void PropagateChildren(Entity parent, const glm::mat4& parentWorld) {
        auto it = mChildren.find(parent);
        if (it == mChildren.end()) return;
        for (Entity child : it->second) {
            auto* ct = mTransforms.Get(child);
            if (!ct) continue;
            ct->worldMatrix = parentWorld * ct->GetLocalMatrix();
            ct->dirty = false;
            PropagateChildren(child, ct->worldMatrix);
        }
    }

    uint32_t            mNextId = 0;
    std::vector<Entity> mFreeList;
    std::set<Entity>    mAlive;

    ComponentPool<TransformComponent> mTransforms;
    ComponentPool<MeshComponent>      mMeshes;
    ComponentPool<MaterialComponent>  mMaterials;
    ComponentPool<LightComponent>     mLights;

    std::unordered_map<Entity, std::vector<Entity>> mChildren;
};
