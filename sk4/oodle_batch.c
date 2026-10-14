/* oodle_batch - decompress many Oodle blocks in one wine invocation.
 *
 * skate. ships its own oo2core_9_win64.dll; that is the exact codec version the
 * game packed with, so we call it rather than guessing at a reimplementation.
 * The DLL is a Windows PE, so this shim is cross-compiled for Win64 and run
 * under wine. Batching matters: per-block process startup dominates otherwise
 * (a single level has tens of thousands of blocks).
 *
 * Build:  see tools/build_oodle_batch.sh
 * Usage:  oodle_batch.exe <oodle.dll> <job> <out>
 *
 * Job file (little-endian):
 *   u32 magic 'OJOB'   u32 count
 *   count * { u64 compSize, u64 rawSize }
 *   payloads, concatenated, in order
 * Output: the raw blocks, concatenated, in order.
 */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef intptr_t (*OodleLZ_Decompress_t)(
    const void *compBuf, intptr_t compBufSize, void *rawBuf, intptr_t rawLen,
    int fuzzSafe, int checkCRC, int verbosity,
    void *decBufBase, intptr_t decBufSize,
    void *fpCallback, void *callbackUserData,
    void *decoderMemory, intptr_t decoderMemorySize,
    int threadPhase);

#define FUZZ_SAFE_YES 1
#define CHECK_CRC_NO  0
#define UNTHREADED    3

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <dll> <job> <out>\n", argv[0]); return 2; }

    HMODULE h = LoadLibraryA(argv[1]);
    if (!h) { fprintf(stderr, "LoadLibrary(%s) failed: %lu\n", argv[1], GetLastError()); return 3; }
    OodleLZ_Decompress_t decompress =
        (OodleLZ_Decompress_t)(void *)GetProcAddress(h, "OodleLZ_Decompress");
    if (!decompress) { fprintf(stderr, "no OodleLZ_Decompress export\n"); return 3; }

    FILE *jf = fopen(argv[2], "rb");
    if (!jf) { fprintf(stderr, "open job failed\n"); return 4; }
    uint32_t magic = 0, count = 0;
    if (fread(&magic, 4, 1, jf) != 1 || magic != 0x424F4A4Fu) {   /* 'OJOB' */
        fprintf(stderr, "bad job magic\n"); return 4;
    }
    if (fread(&count, 4, 1, jf) != 1) { fprintf(stderr, "short job\n"); return 4; }

    uint64_t *sizes = malloc((size_t)count * 16);
    if (!sizes) { fprintf(stderr, "oom sizes\n"); return 5; }
    if (fread(sizes, 16, count, jf) != count) { fprintf(stderr, "short table\n"); return 4; }

    FILE *of = fopen(argv[3], "wb");
    if (!of) { fprintf(stderr, "open out failed\n"); return 4; }

    /* One pair of buffers, grown as needed, reused across blocks. */
    size_t cap_c = 0, cap_r = 0;
    unsigned char *cbuf = NULL, *rbuf = NULL;
    int failed = 0;

    for (uint32_t i = 0; i < count; i++) {
        uint64_t cs = sizes[i * 2], rs = sizes[i * 2 + 1];
        if (cs > cap_c) { cap_c = (size_t)cs; cbuf = realloc(cbuf, cap_c); }
        if (rs > cap_r) { cap_r = (size_t)rs; rbuf = realloc(rbuf, cap_r); }
        if ((cs && !cbuf) || (rs && !rbuf)) { fprintf(stderr, "oom block %u\n", i); return 5; }
        if (fread(cbuf, 1, (size_t)cs, jf) != cs) { fprintf(stderr, "short payload %u\n", i); return 4; }

        intptr_t got = decompress(cbuf, (intptr_t)cs, rbuf, (intptr_t)rs,
                                  FUZZ_SAFE_YES, CHECK_CRC_NO, 0,
                                  NULL, 0, NULL, NULL, NULL, 0, UNTHREADED);
        if (got != (intptr_t)rs) {
            /* Report and keep going: one bad block should not cost the whole batch. */
            fprintf(stderr, "block %u: got %lld want %llu\n",
                    i, (long long)got, (unsigned long long)rs);
            failed++;
            memset(rbuf, 0, (size_t)rs);
        }
        fwrite(rbuf, 1, (size_t)rs, of);
    }
    fclose(of); fclose(jf);
    if (failed) { fprintf(stderr, "%d/%u blocks failed\n", failed, count); return 1; }
    printf("ok %u blocks\n", count);
    return 0;
}
