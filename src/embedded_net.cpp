// the net baked into the binary. build with -DEMBED_NET -DEVALFILE=\"nets/net.nnue\"
// incbin.h: https://github.com/graphitemaster/incbin (single header, gcc / clang)
#include "incbin.h"
#include <cstddef>

#ifndef EVALFILE
#define EVALFILE "released_nets/netv4.bin"
#endif

INCBIN(Net, EVALFILE);               // gNetData (bytes), gNetEnd, gNetSize

const unsigned char* embeddedNetData() { return gNetData; }
size_t embeddedNetSize() { return (size_t)gNetSize; }