#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

class Window;

class InputManager {
public:
    enum class Action : uint32_t {
        MoveForward, MoveBackward, MoveLeft, MoveRight,
        MoveUp, MoveDown,
        SpeedBoost,
        ToggleUI,
        Quit,
        Count
    };

    void Initialize();
    void Update(const Window& window);

    bool IsActive(Action action)   const;
    bool WasPressed(Action action) const;

    bool IsMouseButtonDown(int button) const;

    float GetMouseDX()  const { return mMouseDX; }
    float GetMouseDY()  const { return mMouseDY; }
    float GetScrollDY() const { return mScrollDY; }

    // Gamepad
    bool  IsGamepadConnected() const { return mGamepadConnected; }
    float GetGamepadAxis(int axis) const;
    bool  IsGamepadButtonDown(int button) const;

    // Rebindable mappings
    void  SetKeyBinding(Action action, int glfwKey);
    int   GetKeyBinding(Action action) const;
    void  LoadBindings(const std::string& path);
    void  SaveBindings(const std::string& path) const;

    static const char* ActionName(Action action);

private:
    static constexpr uint32_t kCount        = static_cast<uint32_t>(Action::Count);
    static constexpr uint32_t kMouseButtons = 3;

    int  mKeyBinding[kCount]{};
    bool mCurrent[kCount]{};
    bool mPrevious[kCount]{};
    bool mMouseButtons[kMouseButtons]{};
    float mMouseDX = 0, mMouseDY = 0, mScrollDY = 0;

    bool  mGamepadConnected = false;
    float mGamepadAxes[6]{};
    bool  mGamepadButtons[16]{};
};
