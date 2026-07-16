#include "rle.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
    char buf[128];

    /* Basic encode. */
    assert(rle_encode("aaabbc", buf, sizeof buf) == 6);
    assert(strcmp(buf, "3a2b1c") == 0);

    /* Single characters. */
    assert(rle_encode("abc", buf, sizeof buf) == 6);
    assert(strcmp(buf, "1a1b1c") == 0);

    /* Runs longer than 9 split into chunks of at most 9. */
    assert(rle_encode("aaaaaaaaaaaa", buf, sizeof buf) == 4); /* 12 a's */
    assert(strcmp(buf, "9a3a") == 0);

    /* Empty input. */
    assert(rle_encode("", buf, sizeof buf) == 0);
    assert(strcmp(buf, "") == 0);

    /* Encode buffer too small: "1a1b" + NUL needs 5. */
    assert(rle_encode("ab", buf, 4) == -1);
    assert(rle_encode("ab", buf, 5) == 4);

    /* Basic decode. */
    assert(rle_decode("3a2b1c", buf, sizeof buf) == 6);
    assert(strcmp(buf, "aaabbc") == 0);

    /* Decode empty. */
    assert(rle_decode("", buf, sizeof buf) == 0);
    assert(strcmp(buf, "") == 0);

    /* Malformed inputs. */
    assert(rle_decode("3", buf, sizeof buf) == -1);   /* odd length */
    assert(rle_decode("a3", buf, sizeof buf) == -1);  /* count not a digit */
    assert(rle_decode("0a", buf, sizeof buf) == -1);  /* zero count */

    /* Decode buffer too small: "aaa" + NUL needs 4. */
    assert(rle_decode("3a", buf, 3) == -1);
    assert(rle_decode("3a", buf, 4) == 3);

    /* Round trip. */
    {
        const char *original = "wwwwwwwwwwwwbbbxyyyyyzzz";
        char enc[128];
        char dec[128];
        int n = rle_encode(original, enc, sizeof enc);
        assert(n > 0);
        assert(rle_decode(enc, dec, sizeof dec) == (int)strlen(original));
        assert(strcmp(dec, original) == 0);
    }

    printf("ALL TESTS PASSED\n");
    return 0;
}
