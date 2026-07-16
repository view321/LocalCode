'use strict';

/**
 * Turn a title into a URL slug.
 *
 * Rules:
 *  - the result is lowercase
 *  - letters with diacritics are folded to their base letter (e.g. "é" -> "e",
 *    "û" -> "u"); hint: NFD normalization separates combining marks
 *  - every maximal run of characters other than a-z and 0-9 becomes a single "-"
 *  - the result has no leading or trailing "-"
 *  - an input with nothing usable (e.g. "!!!") returns ""
 *
 * slugify("Hello, World!") === "hello-world"
 * slugify("Crème Brûlée")  === "creme-brulee"
 *
 * @param {string} title
 * @returns {string}
 */
function slugify(title) {
  return String(title)
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

module.exports = { slugify };
