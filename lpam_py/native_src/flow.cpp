#include <stdexcept>
#include <cstring>
#include <typeinfo>
#include "BPFlow.h"
extern "C" __declspec(dllexport) int discrete_flow(
 const unsigned char* a,const unsigned char* b,int h,int w,
 int* ox,int* oy,int* wx,int* wy,double alpha,double d,double gamma,
 int iterations,int hierarchy,int window,double* output) {
    try {
        BPFlow bp;
        bp.setDisplay(false);
        bp.LoadImages(w,h,128,a,w,h,b);
        bp.setPara(alpha,d);
        bp.setHomogeneousMRF(window);
        bp.LoadOffset(ox,oy);
        bp.LoadWinSize(wx,wy);
        bp.ComputeDataTerm(); bp.ComputeRangeTerm(gamma);
        bp.MessagePassing(iterations,hierarchy);
        bp.ComputeVelocity();
        memcpy(output,bp.flow().data(),sizeof(double)*h*w*2);
        return 0;
    } catch (const std::bad_alloc&) { return 2; }
      catch (...) { return 3; }
}
