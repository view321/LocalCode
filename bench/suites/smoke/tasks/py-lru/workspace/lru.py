"""A fixed-capacity least-recently-used (LRU) cache."""


class LruCache:
    """Stores up to `capacity` items; when full, the least recently *used*
    entry is evicted first. Both get() and put() count as a use of the key.
    """

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._items = {}   # key -> value
        self._order = []   # keys, least recently used first

    def get(self, key, default=None):
        """Return the value for `key` (refreshing its recency), or `default`."""
        if key not in self._items:
            return default
        return self._items[key]

    def put(self, key, value):
        """Insert or update `key`, evicting the least recently used entry if
        the cache is full."""
        if key in self._items:
            self._items[key] = value
            self._order.remove(key)
            self._order.append(key)
            return
        if len(self._items) >= self.capacity:
            oldest = self._order.pop()
            del self._items[oldest]
        self._items[key] = value
        self._order.append(key)

    def __len__(self):
        return len(self._items)
