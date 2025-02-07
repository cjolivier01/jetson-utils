#pragma once
namespace hm {
namespace utils {


#define SHOW_IMAGE(_mat$)                                                \
  do {                                                                   \
    show_image(std::string(#_mat$), (_mat$)->download(), /*wait=*/true); \
  } while (false)

#define SHOW_SCALED(_mat$, _scale$)                                                       \
  do {                                                                                    \
    displayScaledImage(std::string(#_mat$), (_mat$)->download(), _scale$, /*wait=*/true); \
  } while (false)

#define SHOW_SMALL(_mat$)     \
  do {                        \
    SHOW_SCALED(_mat$, 0.05); \
  } while (false)

}
}
