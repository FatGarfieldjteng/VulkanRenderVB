#pragma once

#include <cstdint>

class Window;

class InputManager {
public:
    enum class Action : uint32_t {
        MoveForward, MoveBackward, MoveLeft, MoveRight,
        MoveUp, MoveDown,
        SpeedBoost,
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

private:
    static constexpr uint32_t kCount       = static_cast<uint32_t>(Action::Count);
    static constexpr uint32_t kMouseButtons = 3; // left, right, middle

    int  mKeyBinding[kCount]{};
    bool mCurrent[kCount]{};
    bool mPrevious[kCount]{};
    bool mMouseButtons[kMouseButtons]{};
    float mMouseDX = 0, mMouseDY = 0, mScrollDY = 0;
};
