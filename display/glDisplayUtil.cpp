#include <cstdint>
#include <vector>

class glDisplay;

//--------------------------------------------------------------
std::vector<glDisplay*> gDisplays;

glDisplay* glGetDisplay(uint32_t display) {
  if (display >= gDisplays.size()) return NULL;

  return gDisplays[display];
}

uint32_t glGetNumDisplays() { return gDisplays.size(); }
//--------------------------------------------------------------
