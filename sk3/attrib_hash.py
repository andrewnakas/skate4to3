"""Skate 3's 64-bit name hash (vault/AttribSys keys, DepN hashes, locator ids).

The function is Bob Jenkins' lookup8 `hash()` (64-bit, 24-byte blocks, little-endian
words, length added to c before the tail), with the start state a = b = 0xABCDEF0011223344,
c = 0x9E3779B97F4A7C13 (golden ratio). The result is c. Bytes are hashed as given: no case
folding, no prefix, no terminator (`Z_summercamp_spot8_Start` keys with its capitals).

How found: the retail .text (recomp's machine-lifted C++, generated/skate3_recomp.70.cpp,
guest function just before 0x82B73B0C) builds two 64-bit constants with lis/ori/rldimi:
0xABCDEF00<<32|0x11223344 into the a and b registers and 0x9E3779B9<<32|0x7F4A7C13 into c,
then runs a 24-byte block loop (r4 = length, divwu by 24) of byte loads assembled
little-endian and the lookup8 mix shift ladder (43, 9, 8, 38, 23, 5, 35, 49, 11, 12, 18, 22).
Earlier attempts had tried the level 0x11223344ABCDEF00 and the golden ratio in a/b, not
this word order / register assignment. Verified on 1012/1012 DepN file-name pairs from the
retail db.big vaults, `challenge_local_data` = 0x46A56ADDF284C381, and every DLC locator id
(.loc/.rx2 file ids = hash of the locator name). See specs/sk3_packaging.md.
"""

M64 = (1 << 64) - 1
INIT_AB = 0xABCDEF0011223344
INIT_C = 0x9E3779B97F4A7C13


def _mix(a, b, c):
    a = (a - b - c) & M64; a ^= c >> 43
    b = (b - c - a) & M64; b ^= (a << 9) & M64
    c = (c - a - b) & M64; c ^= b >> 8
    a = (a - b - c) & M64; a ^= c >> 38
    b = (b - c - a) & M64; b ^= (a << 23) & M64
    c = (c - a - b) & M64; c ^= b >> 5
    a = (a - b - c) & M64; a ^= c >> 35
    b = (b - c - a) & M64; b ^= (a << 49) & M64
    c = (c - a - b) & M64; c ^= b >> 11
    a = (a - b - c) & M64; a ^= c >> 12
    b = (b - c - a) & M64; b ^= (a << 18) & M64
    c = (c - a - b) & M64; c ^= b >> 22
    return a, b, c


def hash_bytes(k: bytes) -> int:
    """64-bit key of a byte string."""
    n = len(k)
    a = b = INIT_AB
    c = INIT_C
    i = 0
    while n - i >= 24:
        a = (a + int.from_bytes(k[i:i + 8], 'little')) & M64
        b = (b + int.from_bytes(k[i + 8:i + 16], 'little')) & M64
        c = (c + int.from_bytes(k[i + 16:i + 24], 'little')) & M64
        a, b, c = _mix(a, b, c)
        i += 24
    c = (c + n) & M64
    for j, ch in enumerate(k[i:]):
        if j < 8:
            a = (a + (ch << (8 * j))) & M64
        elif j < 16:
            b = (b + (ch << (8 * (j - 8)))) & M64
        else:  # c's low byte holds the length
            c = (c + (ch << (8 * (j - 15)))) & M64
    return _mix(a, b, c)[2]


def attrib_key(name: str) -> int:
    """64-bit key of a name (vault record/class/collection, DepN file, locator)."""
    return hash_bytes(name.encode('latin1'))
