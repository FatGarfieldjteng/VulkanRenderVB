#pragma once

#ifdef VRB_DEBUG
    #include <spdlog/spdlog.h>

    #define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
    #define LOG_INFO(...)  spdlog::info(__VA_ARGS__)
    #define LOG_WARN(...)  spdlog::warn(__VA_ARGS__)
    #define LOG_ERROR(...) spdlog::error(__VA_ARGS__)

    namespace Logger {
        inline void Initialize() {
            spdlog::set_level(spdlog::level::trace);
            spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
        }
    }
#else
    #define LOG_TRACE(...) ((void)0)
    #define LOG_INFO(...)  ((void)0)
    #define LOG_WARN(...)  ((void)0)
    #define LOG_ERROR(...) ((void)0)

    namespace Logger {
        inline void Initialize() {}
    }
#endif

#ifdef VRB_DEBUG
    #define VK_CHECK(result)                                                          \
        do {                                                                          \
            VkResult _res = (result);                                                 \
            if (_res != VK_SUCCESS) {                                                 \
                LOG_ERROR("VkResult {} at {}:{}", static_cast<int>(_res), __FILE__, __LINE__); \
            }                                                                         \
        } while (0)
#else
    #define VK_CHECK(result) (result)
#endif
