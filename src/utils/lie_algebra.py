import jax
import jax.numpy as jnp

from collections import defaultdict

# =============================================================================
# Logsignature dimension
# =============================================================================

def mobius_function(n: int) -> int:
    """
    Compute the Möbius function μ(n).

    The Möbius function is defined as:
      - μ(1) = 1.
      - μ(n) = 0 if n has a squared prime factor.
      - μ(n) = (-1)^r if n is the product of r distinct primes.

    Args:
        n (int): The integer for which to compute μ(n).

    Returns:
        int: The value of μ(n).
    """
    if n == 1:
        return 1
    
    # Use trial division to factorize n.
    p = 2
    count = 0
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            count += 1
            temp //= p
            # If p divides n more than once, then n is not squarefree.
            if temp % p == 0:
                return 0
        p += 1
    if temp > 1:
        count += 1
    return -1 if count % 2 == 1 else 1


def get_logsig_dimension(order : int, dim : int) -> int:
    """
    Compute Witt's formula given by

        sum_{l=1}^m (1/l) * sum_{x | l} mu(l/x) * d^x

    where the inner sum is taken over all positive divisors x of l and
    mu is the Möbius function.

    Args:
        m (int): Upper limit for l in the outer summation.
        d (int): The base used in the exponentiation d^x.

    Returns:
        (int) The result of the formula as a tensor (dtype=torch.float64).
    """
    total = 0.0
    
    for length in range(1, order + 1):
        inner_sum = 0.0

        for x in range(1, length + 1):
            if length % x == 0:
                inner_sum += mobius_function(length // x) * (dim ** x)

        total += inner_sum / length

    return int(total)

# =============================================================================
# Lie algebra - Lyndon words helpers
# =============================================================================

def get_lyndon_words(order: int, dim: int) -> list[list[list[int]]]:
    """
    Generate Lyndon words of lengths 1 to `order` over a `dim`-symbol alphabet,
    returning the results as lists of lists of integers.

    Args:
        order (int): Maximum length of the Lyndon words.
        dim (int): Size of the alphabet (symbols 0 to dim-1).

    Returns:
        list[list[list[int]]]: A list with `order` elements, where the i-th element is a list 
        containing all Lyndon words of length i+1. Each Lyndon word is represented as a list of integers.
    """
    list_of_lyndon_words = defaultdict(list)
    word = [-1]

    while word:
        word[-1] += 1
        m = len(word)
        
        # Save a copy of the current word
        list_of_lyndon_words[m - 1].append(word.copy())
        
        # Extend the word until it reaches the desired order
        while len(word) < order:
            word.append(word[-m])
        
        # Backtrack if the last element reached the maximum symbol
        while word and word[-1] == dim - 1:
            word.pop()

    # Return the list of words for each length from 1 to order.
    return [list_of_lyndon_words[i] for i in range(order)]


def get_lyndon_words_dim(words : list[list]) -> int:
    """
    Compute the dimension of the Lie algebra given a list of Lyndon words.

    Args:
        words (list[list]): A list of Lyndon words, where each word is a list of integers.

    Returns:
        int: CHANGE
    """
    return [len(level) for level in words]


def is_lyndon_word(word: list[int]) -> bool:
    """
    Check if a word (list of integers) is a Lyndon word.
    
    A word is a Lyndon word if it is strictly lexicographically
    smaller than all of its nontrivial rotations.
    
    Args:
        word (list[int]): The word to check.
    
    Returns:
        bool: True if the word is a Lyndon word, False otherwise.
    """
    n = len(word)
    for i in range(1, n):
        # Generate the i-th rotation of the word.
        rotated = word[i:] + word[:i]
        if word >= rotated:
            return False
    return True


def split_lyndon_word(word : list[int]) -> tuple[list[int], list[int]]:
    """
    Given a Lyndon word (as a list of integers or a torch.Tensor) of length > 1,
    return a pair (u, v) of Lyndon words such that their concatenation equals the input word.
    
    Args:
        word (list[int]): The input Lyndon word.
    
    Returns:
        tuple[list[int], list[int]]: Two Lyndon words u and v such that u + v equals the input word.
    
    Raises:
        ValueError: If the input word has length less than 2 or if no valid split is found.
    """
    # If the input is a torch.Tensor, convert it to a Python list.
    if hasattr(word, "tolist"):
        _word = word.tolist()
    else:
        _word = word

    if len(_word) < 2:
        raise ValueError("A Lyndon word of length 1 cannot be split further.")

    n = len(_word)
    
    # Iterate over possible split points.
    for i in range(n - 1, 0, -1):
        prefix = _word[:i]
        suffix = _word[i:]
        if is_lyndon_word(prefix) and is_lyndon_word(suffix):
            return prefix, suffix

    # According to our assumption, this point should never be reached.
    raise ValueError("No valid split found, which should not happen for a Lyndon word of length > 1.")


# =============================================================================
# Lie algebra - Lie matrices
# =============================================================================

@jax.jit
def _commutator_batch(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Compute [A,B] for a batch: A,B: (K, M, M) -> (K, M, M)."""
    return A @ B - B @ A

def get_lie_matrices_from_lyndon_words(words, matrices: jnp.ndarray) -> jnp.ndarray:
    """
    Build Lie matrices L(w) level-by-level from Lyndon words.
    words[0] are the letters; `matrices` has shape (d, M, M) for those letters.
    Returns all levels concatenated: (W_total, M, M).
    """
    levels = [matrices]  # level 0

    # For fast lookup: per level, map word -> index
    level_maps = [{tuple(w): i for i, w in enumerate(level)} for level in words]

    for L in range(1, len(words)):
        # Concatenate all previous levels for simple global indexing
        bank = jnp.concatenate(levels, axis=0)  # (W_prev_total, M, M)

        # Compute offsets of each previous level in the bank
        sizes = [lvl.shape[0] for lvl in levels]
        offsets = [0]
        for s in sizes[:-1]:
            offsets.append(offsets[-1] + s)

        # Gather pairs (A,B) for this level
        idxA, idxB = [], []
        for w in words[L]:
            w1, w2 = split_lyndon_word(w)
            l1, i1 = len(w1) - 1, level_maps[len(w1) - 1][tuple(w1)]
            l2, i2 = len(w2) - 1, level_maps[len(w2) - 1][tuple(w2)]
            idxA.append(offsets[l1] + i1)
            idxB.append(offsets[l2] + i2)

        A = bank[jnp.asarray(idxA, jnp.int32)]  # (K, M, M)
        B = bank[jnp.asarray(idxB, jnp.int32)]  # (K, M, M)
        levels.append(_commutator_batch(A, B))  # (K, M, M)

    return jnp.concatenate(levels, axis=0)

def get_lie_matrices(matrices: jnp.ndarray, order: int) -> jnp.ndarray:
    d = matrices.shape[0]
    words = get_lyndon_words(order, d)
    return get_lie_matrices_from_lyndon_words(words, matrices)