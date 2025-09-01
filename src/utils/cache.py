class Cache:
    """
    A simple in-memory cache.
    """
    
    def __init__(self):
        self.cache = {}

    def get(self, key):
        """
        Retrieve an item from the cache.
        """
        return self.cache.get(key, None)

    def set(self, key, value):
        """
        Store an item in the cache.
        """
        self.cache[key] = value

    def has(self, key):
        """
        Check if a key exists in the cache.
        """
        return key in self.cache
    
    def remove(self, key):
        """
        Remove an item from the cache.
        """
        try:
            del self.cache[key]
        except KeyError:
            pass

    def clear(self):
        """
        Clear the entire cache.
        """
        self.cache.clear()

