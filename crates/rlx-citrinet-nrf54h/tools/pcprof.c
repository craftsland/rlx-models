/* Instruction profile for QEMU: instructions executed per translation block,
   keyed by the block's first PC, written to the file named by `out=` (default
   pcprof.txt) as "pc count" lines. Map PCs to functions with the firmware's
   symbol table (`tools/pcprof.py`).

   Build (macOS, Homebrew QEMU):
     cc -O2 -shared -fPIC -I"$(brew --prefix qemu)/include" \
        $(pkg-config --cflags glib-2.0) -undefined dynamic_lookup \
        -o libpcprof.dylib pcprof.c */
#include <qemu-plugin.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

struct block {
    uint64_t pc;
    uint64_t insns;
    uint64_t execs;
};

#define MAX_BLOCKS (1 << 20)
static struct block *blocks;
static size_t n_blocks;
static const char *out_path = "pcprof.txt";

static void tb_exec(unsigned int cpu, void *udata) {
    ((struct block *)udata)->execs++;
}

static void tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb) {
    if (n_blocks == MAX_BLOCKS) {
        return;
    }
    struct block *b = &blocks[n_blocks++];
    b->pc = qemu_plugin_tb_vaddr(tb);
    b->insns = qemu_plugin_tb_n_insns(tb);
    b->execs = 0;
    qemu_plugin_register_vcpu_tb_exec_cb(tb, tb_exec, QEMU_PLUGIN_CB_NO_REGS, b);
}

static void at_exit(qemu_plugin_id_t id, void *p) {
    FILE *f = fopen(out_path, "w");
    if (!f) {
        return;
    }
    for (size_t i = 0; i < n_blocks; i++) {
        if (blocks[i].execs) {
            fprintf(f, "%" PRIx64 " %" PRIu64 "\n", blocks[i].pc,
                    blocks[i].insns * blocks[i].execs);
        }
    }
    fclose(f);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,
                                           const qemu_info_t *info,
                                           int argc, char **argv) {
    for (int i = 0; i < argc; i++) {
        if (strncmp(argv[i], "out=", 4) == 0) {
            out_path = argv[i] + 4;
        }
    }
    blocks = calloc(MAX_BLOCKS, sizeof *blocks);
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans);
    qemu_plugin_register_atexit_cb(id, at_exit, NULL);
    return 0;
}
