#include "rle.h"

#include <string.h>

int rle_encode(const char *in, char *out, size_t out_cap)
{
    size_t len = strlen(in);
    size_t i = 0;
    size_t w = 0;

    while (i < len) {
        char c = in[i];
        size_t run = 1;
        while (i + run < len && in[i + run] == c && run < 9)
            run++;
        if (w + 2 + 1 > out_cap)
            return -1;
        out[w++] = (char)('0' + (int)run);
        out[w++] = c;
        i += run;
    }
    if (w + 1 > out_cap)
        return -1;
    out[w] = '\0';
    return (int)w;
}

int rle_decode(const char *in, char *out, size_t out_cap)
{
    size_t len = strlen(in);
    size_t w = 0;

    if (len % 2 != 0)
        return -1;
    for (size_t i = 0; i < len; i += 2) {
        char d = in[i];
        if (d < '1' || d > '9')
            return -1;
        size_t run = (size_t)(d - '0');
        if (w + run + 1 > out_cap)
            return -1;
        memset(out + w, in[i + 1], run);
        w += run;
    }
    if (w + 1 > out_cap)
        return -1;
    out[w] = '\0';
    return (int)w;
}
