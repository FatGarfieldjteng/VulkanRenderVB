#pragma once

#include <glm/glm.hpp>
#include <limits>

struct AABB {
    glm::vec3 min{std::numeric_limits<float>::max()};
    glm::vec3 max{std::numeric_limits<float>::lowest()};

    void Include(const glm::vec3& point) {
        min = glm::min(min, point);
        max = glm::max(max, point);
    }

    void Include(const AABB& other) {
        min = glm::min(min, other.min);
        max = glm::max(max, other.max);
    }

    glm::vec3 Extent() const { return max - min; }
    glm::vec3 Center() const { return (min + max) * 0.5f; }

    float Volume() const {
        glm::vec3 ext = Extent();
        return ext.x * ext.y * ext.z;
    }

    bool Valid() const {
        return min.x <= max.x && min.y <= max.y && min.z <= max.z;
    }
};
