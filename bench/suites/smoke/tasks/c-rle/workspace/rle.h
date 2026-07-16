#ifndef RLE_H
#define RLE_H

#include <stddef.h>

/* Run-length encode the NUL-terminated string `in` into `out`.
 *
 * Format: each maximal run of a repeated character c is written as the digit
 * of its length followed by c. Runs longer than 9 are split into chunks of at
 * most 9 (e.g. 12 x 'a' -> "9a3a"). The input never contains digit
 * characters.
 *
 * `out_cap` is the capacity of `out` INCLUDING space for the trailing NUL.
 * On success the encoded string is NUL-terminated and the number of
 * characters written (excluding the NUL) is returned. If `out` is too small,
 * -1 is returned. An empty input writes an empty string and returns 0.
 */
int rle_encode(const char *in, char *out, size_t out_cap);

/* Decode a string produced by rle_encode back into `out`.
 *
 * Returns the number of characters written (excluding the NUL), or -1 when
 * `out` is too small or the input is malformed: odd length, a chunk whose
 * first character is not a digit 1-9 (zero counts are malformed too).
 */
int rle_decode(const char *in, char *out, size_t out_cap);

#endif /* RLE_H */
