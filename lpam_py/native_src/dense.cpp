#include <stdexcept>
#include <cstring>
#include "ImageFeature.h"
extern "C" __declspec(dllexport) int dense_sift(const double* rgb, int h, int w, unsigned char* out) {
    try {
        if (!rgb || !out || h < 1 || w < 1) return 1;
        DImage image(w,h,3);
        memcpy(image.data(),rgb,sizeof(double)*h*w*3);
        UCImage sift;
        ImageFeature::imSIFT(image,sift,3,1,true);
        memcpy(out,sift.data(),size_t(h)*w*128);
        return 0;
    } catch (const std::bad_alloc&) { return 2; }
      catch (...) { return 3; }
}
