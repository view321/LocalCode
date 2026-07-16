#include "ringbuf.h"

#include <stdlib.h>

int rb_init(ringbuf *rb, size_t cap)
{
    if (cap == 0)
        return -1;
    rb->data = malloc(cap);
    if (rb->data == NULL)
        return -1;
    rb->cap = cap;
    rb->head = 0;
    rb->count = 0;
    return 0;
}

void rb_free(ringbuf *rb)
{
    free(rb->data);
    rb->data = NULL;
    rb->cap = 0;
    rb->head = 0;
    rb->count = 0;
}

int rb_push(ringbuf *rb, unsigned char v)
{
    if (rb->count == rb->cap)
        return -1;
    size_t tail = (rb->head + rb->count) % rb->cap;
    rb->data[tail] = v;
    rb->count++;
    return 0;
}

int rb_pop(ringbuf *rb, unsigned char *out)
{
    if (rb->count == 0)
        return -1;
    *out = rb->data[rb->head];
    rb->head = (rb->head + 1) % rb->cap;
    rb->count--;
    return 0;
}

size_t rb_len(const ringbuf *rb)
{
    return rb->count;
}
