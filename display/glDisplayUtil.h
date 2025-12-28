#pragma once

#include <cstdint>

namespace jetson_utils {

struct IGlEventSink {
  /**
   * Event message handler callback for recieving UI messages from a window.
   *
   * Recieves 4 parameters - the event type, a & b message values (see above),
   *                         and a user-specified pointer from registration.
   *
   * Event message handlers should return `true` if the message was
   * handled, or `false` if the message was skipped or not handled.
   *
   * @see glRegisterEvents
   * @see glRemoveEvents
   *
   * @ingroup OpenGL
   */
  typedef bool (*glEventHandler)(uint16_t event, int a, int b, void* user);

  /**
   * Register an event message handler that will be called by ProcessEvents()
   * @param callback function pointer to the event message handler callback
   * @param user optional user-specified pointer that will be passed to all
   *             invocations of this event handler (typically an object)
   */
  virtual void AddEventHandler(glEventHandler callback, void* user = nullptr) = 0;

  /**
   * Remove an event message handler from being called by ProcessEvents()
   * RemoveEventHandler() will search for previously registered event
   * handlers that have the same function pointer and/or user pointer,
   * and remove them for being called again in the future.

   */
  virtual void RemoveEventHandler(glEventHandler callback,
                                  void* user = nullptr) = 0;
};

struct glDisplayBase : public IGlEventSink {};
}  // namespace jetson_utils

std::size_t glAddDisplay(jetson_utils::glDisplayBase* display);
jetson_utils::glDisplayBase* glGetDisplay(uint32_t display);
uint32_t glGetNumDisplays();
bool glRemoveDisplay(jetson_utils::glDisplayBase* display);
