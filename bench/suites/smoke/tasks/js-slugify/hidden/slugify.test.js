'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { slugify } = require('./slugify.js');

test('basic title', () => {
  assert.equal(slugify('Hello, World!'), 'hello-world');
});

test('diacritics fold to base letters', () => {
  assert.equal(slugify('Crème Brûlée'), 'creme-brulee');
  assert.equal(slugify('Ünïcödé'), 'unicode');
});

test('punctuation runs collapse to one dash', () => {
  assert.equal(slugify('a  ---  b!!c'), 'a-b-c');
});

test('no leading or trailing dashes', () => {
  assert.equal(slugify('  --Already--Sluggy--  '), 'already-sluggy');
});

test('digits survive', () => {
  assert.equal(slugify('42 is THE answer'), '42-is-the-answer');
});

test('empty and unusable inputs', () => {
  assert.equal(slugify(''), '');
  assert.equal(slugify('!!! ??? ///'), '');
});

test('already clean input is unchanged', () => {
  assert.equal(slugify('clean-slug-123'), 'clean-slug-123');
});
