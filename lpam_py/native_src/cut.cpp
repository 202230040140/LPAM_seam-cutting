#include <stdexcept>
#include <climits>
#include <cmath>
#include "graph.h"
#include "graph.cpp"
#include "maxflow.cpp"
extern "C" __declspec(dllexport) int graph_cut(
 int n,int m,const double* terminal,const int* u,const int* v,const double* cost,
 unsigned char* labels,double* energy) {
    try {
        Graph<double,double,double> graph(n,m);
        graph.add_node(n);
        for(int i=0;i<n;i++) {
            double a=terminal[2*i],b=terminal[2*i+1];
            // The original MEX uses the same maxflow double capacities.
            graph.add_tweights(i,std::isinf(a)?double(INT_MAX):a,std::isinf(b)?double(INT_MAX):b);
        }
        for(int i=0;i<m;i++) graph.add_edge(u[i],v[i],cost[i],cost[i]);
        *energy=graph.maxflow();
        for(int i=0;i<n;i++) labels[i]=(unsigned char)graph.what_segment(i);
        return 0;
    } catch (const std::bad_alloc&) { return 2; }
      catch (...) { return 3; }
}
