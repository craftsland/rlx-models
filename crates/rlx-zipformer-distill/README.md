# rlx-zipformer-distill

Trains a small streaming Zipformer2, about 10M parameters, **with RLX** (the
`Func` autodiff + optimizer stack) on whatever devices and machines are
available. It is distilled from the 66M
streaming Zipformer2 medium (`just fetch-zipformer-m`), and the result runs
on `rlx-kroko-core`, the `no_std` MCU runtime.

## How

* **Same computation in training and on the MCU.** `model.rs` is
  `rlx-kroko-core`'s chunk loop as an RLX graph:
  - the conv embed with its 3-frame cache, and six stacks at 1/2/4/8/4/2× rate;
  - per-layer key / non-linear / value / conv caches, and compact relative
    positions;
  - chunk-local depthwise convs with edge scales, and BiasNorm and the bypasses.

  It is unrolled over the K = 8 chunks (2.56 s) of a segment from a zero
  state. Checked against the core with the teacher's own config and weights
  (`just zipformer-distill-graph-check`): encoder cosine 0.9989–0.9998 per
  chunk against the core's int8 teacher on JFK, on CPU and Metal.
* **Targets.** The teacher's encoder output (in the joiner's input space), on
  the same segments, computed by `rlx-kroko-core` (`zipformer-distill-prep`),
  and the predictor context at every output frame from the teacher's greedy
  decoding of it (`zipformer-distill-prep --contexts`, `contexts.u16`, a few
  seconds per 20 h).
* **Loss.** The student keeps the teacher's decoder and joiner. The loss is
  the KL divergence from the teacher's joiner token distribution to the
  student's at every frame (both through the teacher's predictor output for
  the teacher's context), plus 0.1 × the squared error to the teacher's
  encoder output over its mean square (`--mse-weight`, `--temperature`).
  Matching the encoder output alone (`--mse-only`, or no `contexts.u16`)
  spends the model's capacity evenly on channels that never change a token.
* **Resuming.** Every save also writes `resume.bin` (steps done, parameters,
  AdamW moments); `--resume` with the same arguments continues exactly.
* **Initialization.** Each student tensor is the teacher's tensor of the same
  name, cut to the student's widths (leading channels, leading heads per
  segment of a fused projection, both GLU halves). The conv embed and stack 0
  match the teacher's shapes exactly.
* **Export.** Parameters carry the names `rlx-kroko`'s exporter reads, so
  saving them, plus the teacher's decoder/joiner and a `config.json`, gives a
  directory for `rlx-kroko`'s `mcu_check --dir` (→ `.rkr`, WER, QEMU).

Student: widths 192,160,192,256,192,160, layers 1,1,2,2,2,1, 4 heads, the
teacher's rates and kernels. 11.05M parameters (encoder + embed) plus the
teacher's 0.8M decoder/joiner. The int8 `.rkr` is 13.3 MB (11.4 MB of it paged
layers), with 1.4 MiB scratch and a 1.8 MiB page buffer.

## Devices and machines

Nothing is configured per machine.

* **Local devices.** Every backend the binary was built with (`FEATURES`) that
  can train is a candidate, one per chip (`rlx_tensor::training_devices`: the
  native GPU API over portable ones, the CPU included, NPUs only when named).
  Each is timed alone and against the speed split
  (`rlx_tensor::parallel::tune`); the fastest layout wins and a device too
  slow to help is dropped. Each device takes its share of every batch
  (`DataParallel`: its own thread, gradients averaged by share, one AdamW step
  on the host — the same update as one device with the whole batch).
  `--devices rocm,cpu` restricts the candidates, `--devices rocm:24,cpu:8`
  fixes the split.
* **Other machines.** `--cluster` makes the training process the coordinator:
  it announces itself on the LAN (`rlx-driver` rendezvous, `WORLD=auto`) and
  trains with every worker that joins within `RENDEZVOUS_WINDOW` seconds
  (default 10), or alone. A worker (`just zipformer-distill-worker`) needs only
  the binary: the coordinator sends the job, the initial weights and each
  step's samples. Each machine tunes its own devices; the global `--batch` is
  split across machines by measured samples/s; gradients are summed
  sample-weighted and every machine applies the same update (the
  coordinator's weights are re-broadcast every 100 steps). `--rendezvous
  tcp://host:port` replaces LAN discovery where broadcast does not reach.
* **Batch size.** The unrolled 8-chunk graph is thousands of small kernels, so
  small batches are launch-bound on GPUs: going from batch 8 to 32 roughly
  doubles training samples/s on an integrated GPU.

Inference throughput on the local devices, picked the same way:
`just zipformer-distill-throughput FEATURES BATCH`.

```bash
just fetch-zipformer-m                  # teacher
just zipformer-distill-graph-check      # RLX graph == rlx-kroko-core
just zipformer-distill-prep             # LibriSpeech train-clean-100 → targets (AUDIO, OUT)
just zipformer-distill-train            # RLX, local devices → .cache/zipformer-s
just zipformer-distill-train DATA metal --cluster   # … plus every worker on the LAN
just zipformer-distill-worker rocm      # on another machine: join it
just zipformer-m-check assets/harness/jfk_16k.wav 8   # (point --dir at .cache/zipformer-s)
```
