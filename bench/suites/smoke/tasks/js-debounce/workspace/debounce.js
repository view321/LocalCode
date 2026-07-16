'use strict';

/**
 * debounce(fn, waitMs) returns a debounced wrapper of `fn`:
 *  - calling the wrapper schedules `fn` to run `waitMs` after the *last* call
 *  - calls arriving inside the window reset the timer (trailing edge only)
 *  - when `fn` finally runs it receives the most recent arguments and the
 *    most recent `this`
 *  - wrapper.cancel() drops any pending call
 *
 * @param {Function} fn
 * @param {number} waitMs
 * @returns {Function & { cancel: () => void }}
 */
function debounce(fn, waitMs) {
  let timer = null;
  let lastArgs = null;

  function debounced(...args) {
    lastArgs = args;
    if (timer !== null) {
      return;
    }
    timer = setTimeout(() => {
      timer = null;
      fn.apply(undefined, lastArgs);
    }, waitMs);
  }

  debounced.cancel = function cancel() {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };

  return debounced;
}

module.exports = { debounce };
