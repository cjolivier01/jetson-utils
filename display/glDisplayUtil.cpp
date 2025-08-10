#include "glDisplayUtil.h"

#include <algorithm>
#include <cstdint>
#include <vector>

//--------------------------------------------------------------
namespace jetson_utils {
std::vector<jetson_utils::glDisplayBase*> gDisplays;
}

size_t glAddDisplay(jetson_utils::glDisplayBase* display) {
  jetson_utils::gDisplays.push_back(display);
  return jetson_utils::gDisplays.size() - 1;
}

jetson_utils::glDisplayBase* glGetDisplay(uint32_t display) {
  if (display >= jetson_utils::gDisplays.size()) return NULL;

  return jetson_utils::gDisplays[display];
}

uint32_t glGetNumDisplays() { return jetson_utils::gDisplays.size(); }

bool glRemoveDisplay(jetson_utils::glDisplayBase* display) {
  auto it = std::find(jetson_utils::gDisplays.begin(),
                      jetson_utils::gDisplays.end(), display);
  if (it != jetson_utils::gDisplays.end()) {
    jetson_utils::gDisplays.erase(it);
    return true;
  }
  return false;
}
//--------------------------------------------------------------
