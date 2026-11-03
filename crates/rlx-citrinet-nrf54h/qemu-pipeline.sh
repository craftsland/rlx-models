#!/usr/bin/env bash
# Run Citrinet as a pipeline over several simulated nRF54H20s: one QEMU
# (MPS2 AN505, Cortex-M33) per stage, chained UART to UART over local TCP.
# Stage 0 transcribes the clip and sends activation frames on UART0; stage i
# receives on UART0 and sends on UART1; the last stage checks ids and encoder
# bits against data/ (the same vectors as the single-chip run).
#
#   ./qemu-pipeline.sh STAGE0.rct STAGE1.rct [...]
#
# Stage files come from `examples/mcu_split.rs --out-dir`. Each QEMU prints
# its instruction count (INSNS). A later stage's count includes the UART
# polling while it waits for input; for compute alone, profile with
# PROFILE=DIR (tools/libpcprof.dylib, one DIR/chip<i>.txt per chip) and fold
# with `tools/pcprof.py`, which reports the polling loop separately.
set -euo pipefail
(( $# >= 2 )) || { echo "usage: $0 STAGE0.rct STAGE1.rct [...]" >&2; exit 2; }
stages=()
for f in "$@"; do
    [ -f "$f" ] || { echo "no such stage file: $f" >&2; exit 2; }
    stages+=("$(cd "$(dirname "$f")" && pwd)/$(basename "$f")")
done
cd "$(dirname "$0")"
plugin=../rlx-ten-vad-mcu/libinsn.dylib
profile="${PROFILE:-}"
[ -z "$profile" ] || mkdir -p "$profile"
elf=target/thumbv8m.main-none-eabihf/release/citrinet-nrf54h
mkdir -p target/pipeline
n=${#stages[@]}
for ((i = 0; i < n; i++)); do
    RCT_MODEL="${stages[$i]}" cargo build --release -q 2> >(grep -v "patch \`\|^help:\|with the dep\|locked\|version. This\|^ *$" >&2)
    cp "$elf" "target/pipeline/stage$i.elf"
done
base=$((20000 + RANDOM % 20000))
pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT
status=0
for ((i = n - 1; i >= 0; i--)); do
    serial=()
    (( i > 0 )) && serial+=(-serial "tcp:127.0.0.1:$((base + i)),server=on,wait=on")
    (( i < n - 1 )) && serial+=(-serial "tcp:127.0.0.1:$((base + i + 1))")
    (( i == 0 )) || true
    qemu-system-arm -machine mps2-an505 -display none -monitor none \
        -semihosting-config enable=on,target=native \
        -plugin "$( [ -n "$profile" ] && echo "tools/libpcprof.dylib,out=$profile/chip$i.txt" || echo "$plugin" )" \
        "${serial[@]}" -kernel "target/pipeline/stage$i.elf" 2>&1 | sed "s/^/[chip $i] /" &
    pids+=($!)
    if (( i > 0 )); then
        for _ in $(seq 100); do nc -z 127.0.0.1 $((base + i)) 2>/dev/null && break; sleep 0.1; done
    fi
done
for p in "${pids[@]}"; do wait "$p" || status=1; done
trap - EXIT
exit $status
