#include "ringbuf.h"

#include <assert.h>
#include <stdio.h>

int main(void)
{
    ringbuf rb;
    unsigned char v = 0;

    /* Init validation. */
    assert(rb_init(&rb, 0) == -1);
    assert(rb_init(&rb, 4) == 0);
    assert(rb_len(&rb) == 0);

    /* Empty pop fails. */
    assert(rb_pop(&rb, &v) == -1);

    /* Simple FIFO. */
    assert(rb_push(&rb, 1) == 0);
    assert(rb_push(&rb, 2) == 0);
    assert(rb_len(&rb) == 2);
    assert(rb_pop(&rb, &v) == 0 && v == 1);
    assert(rb_pop(&rb, &v) == 0 && v == 2);
    assert(rb_pop(&rb, &v) == -1);

    /* Fill completely, then overflow fails. */
    for (unsigned char i = 10; i < 14; i++)
        assert(rb_push(&rb, i) == 0);
    assert(rb_len(&rb) == 4);
    assert(rb_push(&rb, 99) == -1);

    /* Wraparound: free two slots, push two more, order must hold. */
    assert(rb_pop(&rb, &v) == 0 && v == 10);
    assert(rb_pop(&rb, &v) == 0 && v == 11);
    assert(rb_push(&rb, 14) == 0); /* these two writes wrap */
    assert(rb_push(&rb, 15) == 0);
    assert(rb_len(&rb) == 4);
    assert(rb_pop(&rb, &v) == 0 && v == 12);
    assert(rb_pop(&rb, &v) == 0 && v == 13);
    assert(rb_pop(&rb, &v) == 0 && v == 14);
    assert(rb_pop(&rb, &v) == 0 && v == 15);
    assert(rb_len(&rb) == 0);

    /* Long churn across many wraps. */
    {
        unsigned char next_in = 0;
        unsigned char next_out = 0;
        int i;
        for (i = 0; i < 3; i++)
            assert(rb_push(&rb, next_in++) == 0);
        for (i = 0; i < 100; i++) {
            assert(rb_push(&rb, next_in++) == 0);
            assert(rb_pop(&rb, &v) == 0);
            assert(v == next_out++);
        }
        for (i = 0; i < 3; i++) {
            assert(rb_pop(&rb, &v) == 0);
            assert(v == next_out++);
        }
        assert(rb_pop(&rb, &v) == -1);
    }

    rb_free(&rb);

    printf("ALL TESTS PASSED\n");
    return 0;
}
