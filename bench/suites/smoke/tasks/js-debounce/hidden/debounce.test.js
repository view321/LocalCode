'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { debounce } = require('./debounce.js');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

test('fires once, after the quiet period, with the last args', async () => {
  const calls = [];
  const d = debounce((x) => calls.push(x), 120);
  d(1);
  d(2);
  d(3);
  assert.equal(calls.length, 0, 'must not fire synchronously');
  await sleep(320);
  assert.deepEqual(calls, [3]);
});

test('calls inside the window reset the timer (trailing edge)', async () => {
  const stamps = [];
  const start = Date.now();
  const d = debounce(() => stamps.push(Date.now() - start), 400);
  d();               // t=0 — alone, this would fire at ~400
  await sleep(150);
  d();               // t≈150 — must push the deadline to ≈550
  await sleep(300);  // t≈450 — past the FIRST deadline, before the reset one
  assert.equal(stamps.length, 0, 'fired from the first call\'s timer — window was not reset');
  await sleep(250);  // t≈700 — well past the reset deadline
  assert.equal(stamps.length, 1);
  assert.ok(stamps[0] >= 540, `fired at ${stamps[0]}ms, expected >= ~550`);
});

test('preserves `this` of the last call', async () => {
  const seen = [];
  const obj = {
    name: 'receiver',
    hit: debounce(function () {
      seen.push(this && this.name);
    }, 80),
  };
  obj.hit();
  await sleep(220);
  assert.deepEqual(seen, ['receiver']);
});

test('cancel drops the pending call', async () => {
  let count = 0;
  const d = debounce(() => {
    count += 1;
  }, 80);
  d();
  d.cancel();
  await sleep(220);
  assert.equal(count, 0);
  // And the wrapper still works after a cancel.
  d();
  await sleep(220);
  assert.equal(count, 1);
});
