#ifndef RINGBUF_H
#define RINGBUF_H

#include <stddef.h>

/* A fixed-capacity FIFO ring buffer of bytes.
 *
 * Values pushed with rb_push come back out of rb_pop in the same order,
 * across any number of wraparounds. Pushing into a full buffer and popping
 * from an empty one fail without changing state.
 */
typedef struct {
    unsigned char *data;
    size_t cap;   /* max elements stored */
    size_t head;  /* index of the next element to pop */
    size_t count; /* elements currently stored */
} ringbuf;

/* Initialize with capacity `cap` (heap-allocated). Returns 0 on success,
 * -1 when cap is 0 or allocation fails. */
int rb_init(ringbuf *rb, size_t cap);

/* Release the buffer's memory. Safe on an already-freed ringbuf. */
void rb_free(ringbuf *rb);

/* Append a value. Returns 0, or -1 when the buffer is full. */
int rb_push(ringbuf *rb, unsigned char v);

/* Remove the oldest value into *out. Returns 0, or -1 when empty. */
int rb_pop(ringbuf *rb, unsigned char *out);

/* Number of elements currently stored. */
size_t rb_len(const ringbuf *rb);

#endif /* RINGBUF_H */
