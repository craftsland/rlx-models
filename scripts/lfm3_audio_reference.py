#!/usr/bin/env python3
"""Reference outputs for `rlx-lfm3-audio`, from LiquidAI's LFM3-Audio-2B weights.

A self-contained torch port of the released modeling code
(`modeling_lfm3.py`, `modeling_lfm3_audio.py`, `modeling_liquid_codec.py`,
`feature_extraction_lfm3_audio.py`): the same modules, the same masks, the
same RoPE, in float32 over the bf16 checkpoint. It exists because those files
need an unreleased transformers; nothing here is taken from the Rust side.

    python3 scripts/lfm3_audio_reference.py \
        --model weights/lm/lfm3-audio-2b \
        --wav assets/harness/jfk_16k.wav \
        --out crates/rlx-lfm3-audio/tests/fixtures/reference.json

Each stage that can be wrong on its own is recorded on its own: the log-mel
frontend, the audio encoder + projector, one LM forward over a fixed prompt,
greedy ASR, the depth decoder on fixed inputs, greedy TTS frames (codebooks
revealed in index order so both sides are deterministic) and the codec
decoding those frames to 24 kHz audio.
"""

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

torch.set_grad_enabled(False)

AUDIO_TOKEN = 125031
AUDIO_OUT_TOKEN = 125032
AUDIO_START = 125012
TEXT_END = 125014
MIXED_START = 125015
IM_END = 124900
BOS = 124894
EOS_IDS = (IM_END, TEXT_END)
NUM_CODEBOOKS = 12
CODEBOOK_SIZE = 2017
AUDIO_EOS_CODE = 2016


# ---------------------------------------------------------------- primitives


def rms_norm(x, w, eps=1e-5):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def linear(x, w, b=None):
    return F.linear(x, w.float(), None if b is None else b.float())


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def inv_freq(theta, head_dim, partial=1.0):
    """LFM3 zero-tail partial RoPE: the first `partial*head_dim//2` frequencies
    follow the full head dimension's progression, the rest are zero."""
    angles = int(partial * head_dim) // 2
    f = 1.0 / (theta ** (torch.arange(0, 2 * angles, 2, dtype=torch.float) / head_dim))
    return torch.cat((f, torch.zeros(head_dim // 2 - angles)))


def rope_tables(positions, inv):
    freqs = positions[:, None].float() * inv[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


# ---------------------------------------------------------------- LFM trunk


class Trunk:
    """LFM2 / LFM3 backbone over `inputs_embeds` (`[T, H]`).

    `kinds[i]` is `conv`, `full_attention` or `sliding_attention`; `rope` and
    `window` map an attention kind to `(theta, partial)` and a window (`None`
    = unbounded). `lfm2` selects LFM2 tensor names (`q_layernorm`,
    `out_proj`, no output gate); `sandwich` adds the post-operator/post-FFN
    norms.
    """

    def __init__(self, sd, prefix, kinds, heads, kv_heads, head_dim, rope, window,
                 causal=True, sandwich=False, lfm2=False, eps=1e-5):
        self.sd, self.p, self.kinds = sd, prefix, kinds
        self.nh, self.nkv, self.hd = heads, kv_heads, head_dim
        self.rope, self.window, self.causal = rope, window, causal
        self.sandwich, self.lfm2, self.eps = sandwich, lfm2, eps

    def w(self, name):
        return self.sd[f"{self.p}{name}"]

    def mask(self, kind, T):
        q = torch.arange(T)[:, None]
        k = torch.arange(T)[None, :]
        allowed = torch.ones(T, T, dtype=torch.bool)
        if self.causal:
            allowed &= k <= q
            w = self.window.get(kind)
            if w is not None:
                allowed &= k > q - w
        return allowed

    def attention(self, i, x, kind):
        T = x.shape[0]
        lp = f"layers.{i}.self_attn."
        qn, kn, on = ("q_layernorm", "k_layernorm", "out_proj") if self.lfm2 else ("q_norm", "k_norm", "o_proj")
        q = linear(x, self.w(lp + "q_proj.weight")).view(T, self.nh, self.hd)
        k = linear(x, self.w(lp + "k_proj.weight")).view(T, self.nkv, self.hd)
        v = linear(x, self.w(lp + "v_proj.weight")).view(T, self.nkv, self.hd)
        q = rms_norm(q, self.w(lp + qn + ".weight"), self.eps)
        k = rms_norm(k, self.w(lp + kn + ".weight"), self.eps)
        theta, partial = self.rope[kind]
        cos, sin = rope_tables(torch.arange(T), inv_freq(theta, self.hd, partial))
        cos, sin = cos[:, None, :], sin[:, None, :]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) * self.hd ** -0.5
        scores = scores.masked_fill(~self.mask(kind, T)[None], float("-inf"))
        out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
        if not self.lfm2:
            gate = torch.sigmoid(linear(x, self.w(lp + "gate_proj.weight")))
            out = out * gate[:, :, None]
        return linear(out.reshape(T, -1), self.w(lp + on + ".weight"))

    def short_conv(self, i, x):
        T, H = x.shape
        lp = f"layers.{i}.conv."
        bcx = linear(x, self.w(lp + "in_proj.weight"))
        b, c, xx = bcx.chunk(3, dim=-1)
        bx = (b * xx).T[None]  # [1, H, T]
        wconv = self.w(lp + "conv.weight").float()  # [H, 1, k]
        kk = wconv.shape[-1]
        conv = F.conv1d(bx, wconv, padding=kk - 1, groups=H)[..., :T]
        y = c * conv[0].T
        return linear(y, self.w(lp + "out_proj.weight"))

    def ffn(self, i, x):
        lp = f"layers.{i}.feed_forward."
        return linear(F.silu(linear(x, self.w(lp + "w1.weight"))) * linear(x, self.w(lp + "w3.weight")),
                      self.w(lp + "w2.weight"))

    def __call__(self, x):
        x = x.float()
        for i, kind in enumerate(self.kinds):
            res = x
            h = rms_norm(x, self.w(f"layers.{i}.operator_norm.weight"), self.eps)
            h = self.short_conv(i, h) if kind == "conv" else self.attention(i, h, kind)
            if self.sandwich:
                h = rms_norm(h, self.w(f"layers.{i}.post_operator_norm.weight"), self.eps)
            x = res + h
            f = self.ffn(i, rms_norm(x, self.w(f"layers.{i}.ffn_norm.weight"), self.eps))
            if self.sandwich:
                f = rms_norm(f, self.w(f"layers.{i}.post_ffn_norm.weight"), self.eps)
            x = x + f
        return rms_norm(x, self.w("embedding_norm.weight"), self.eps)


# ---------------------------------------------------------------- frontend


def mel_filter_bank_slaney(n_freqs, n_mels, f_min, f_max, sr):
    """`transformers.audio_utils.mel_filter_bank(norm="slaney", mel_scale="slaney")`."""

    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        min_log_hz, min_log_mel, logstep = 1000.0, 15.0, 27.0 / np.log(6.4)
        mel = 3.0 * f / 200.0
        return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) * logstep, mel)

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        min_log_hz, min_log_mel, logstep = 1000.0, 15.0, np.log(6.4) / 27.0
        f = 200.0 * m / 3.0
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f)

    mel_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    filter_freqs = mel_to_hz(mel_pts)
    fft_freqs = np.linspace(0, sr // 2, n_freqs)
    diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    enorm = 2.0 / (filter_freqs[2 : n_mels + 2] - filter_freqs[:n_mels])
    return fb * enorm[None, :]  # [n_freqs, n_mels]


def log_mel(samples):
    n_fft, hop = 400, 160
    n = samples.numel()
    padded = n_fft if n < n_fft else n + (-(n - n_fft) % hop)
    samples = F.pad(samples, (0, padded - n))
    window = torch.hann_window(n_fft)
    spec = torch.stft(samples, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window,
                      center=False, return_complex=True)
    power = spec.abs().square()
    fb = torch.from_numpy(mel_filter_bank_slaney(n_fft // 2 + 1, 128, 0.0, 8000.0, 16000)).float()
    lm = (fb.T @ power).log10().clamp(min=-12.0)
    return (lm + 4.0) / 4.0  # [128, frames]


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2, "16-bit PCM expected"
        sr, ch = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(1)
    assert sr == 16000, f"16 kHz expected, got {sr}"
    return torch.from_numpy(pcm)


# ---------------------------------------------------------------- the model


class Lfm3Audio:
    def __init__(self, model_dir):
        sd = load_file(str(Path(model_dir) / "model.safetensors"))
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        self.sd = sd
        lfm = cfg["lfm"]
        self.lm = Trunk(sd, "lfm.model.", lfm["layer_types"], 16, 4, 128,
                        rope={"full_attention": (1e6, 0.5), "sliding_attention": (1e4, 1.0)},
                        window={"full_attention": None, "sliding_attention": lfm["sliding_window"]},
                        sandwich=True)
        enc = cfg["encoder"]["lfm"]
        self.encoder = Trunk(sd, "encoder.lfm.lfm.", enc["layer_types"], 8, 4, 96,
                             rope={"full_attention": (1e6, 1.0)},
                             window={"full_attention": cfg["encoder"]["sliding_window"]},
                             lfm2=True)
        self.depth = Trunk(sd, "depthformer.model.", ["full_attention"] * 3, 32, 8, 32,
                           rope={"full_attention": (1e6, 1.0)}, window={"full_attention": None},
                           causal=False)
        self.embed = sd["lfm.model.embed_tokens.weight"].float()

    # ---- audio in
    def downsample(self, mel):
        sd = self.sd
        p = "encoder.downsample."

        def conv(x, name, groups=1, stride=2, pad=1):
            return F.conv2d(x, sd[p + name + ".weight"].float(), sd[p + name + ".bias"].float(),
                            stride=stride, padding=pad, groups=groups)

        x = mel[None, None]  # [1, 1, 128, T]
        x = F.silu(conv(x, "conv_in"))
        x = F.silu(conv(conv(x, "conv_mid", groups=256), "lin_mid", stride=1, pad=0))
        x = F.silu(conv(conv(x, "conv_end", groups=256), "lin_end", stride=1, pad=0))
        x = x.flatten(1, 2)  # [1, 256*16, T']
        x = F.conv1d(x, sd[p + "lin_out.weight"].float(), sd[p + "lin_out.bias"].float())
        return x[0].T  # [T', 768]

    def audio_features(self, mel):
        enc = self.encoder(self.downsample(mel))
        sd = self.sd
        h = F.layer_norm(enc, (768,), sd["mlp.model.0.weight"].float(), sd["mlp.model.0.bias"].float(), 1e-5)
        h = F.gelu(linear(h, sd["mlp.model.1.weight"], sd["mlp.model.1.bias"]))
        return linear(h, sd["mlp.model.3.weight"], sd["mlp.model.3.bias"])

    # ---- speech out
    def frame_embed(self, codes):
        offs = torch.arange(NUM_CODEBOOKS) * CODEBOOK_SIZE
        return self.sd["codebook_embedding.weight"].float()[codes + offs].sum(0)

    def depth_logits(self, hidden, codes, masked):
        sd = self.sd
        din = linear(hidden, sd["depth_linear.weight"], sd["depth_linear.bias"]).view(NUM_CODEBOOKS, 1024)
        emb = torch.stack([sd[f"depth_embeddings.{i}.embedding.weight"].float()[codes[i]]
                           for i in range(NUM_CODEBOOKS)])
        din = din + torch.where(masked[:, None], sd["mask_embedding"].float()[None], emb)
        out = self.depth(din)
        return torch.stack([
            linear(rms_norm(out[i], sd[f"depth_embeddings.{i}.embedding_norm.weight"]),
                   sd[f"depth_embeddings.{i}.embedding.weight"])
            for i in range(NUM_CODEBOOKS)
        ])

    def greedy_frame(self, hidden, steps=(4, 4, 4)):
        """`sample_audio_frame` at temperature 0 with codebooks revealed in index order."""
        codes = torch.zeros(NUM_CODEBOOKS, dtype=torch.long)
        revealed = torch.zeros(NUM_CODEBOOKS, dtype=torch.bool)
        order = torch.arange(NUM_CODEBOOKS)
        for pick in order.split(list(steps)):
            logits = self.depth_logits(hidden, codes, ~revealed)
            for i in pick.tolist():
                codes[i] = int(logits[i].argmax())
            revealed[pick] = True
        return codes

    # ---- LM
    def hidden(self, embeds):
        return self.lm(embeds)

    def logits(self, hidden):
        return hidden @ self.embed.T


class Codec:
    def __init__(self, model_dir):
        self.sd = load_file(str(Path(model_dir) / "audio_tokenizer" / "model.safetensors"))
        kinds = ["sliding_attention" if i in (2, 4, 6) else "conv" for i in range(8)]
        self.decoder = Trunk(self.sd, "decoder.lfm.", kinds, 8, 2, 64,
                             rope={"sliding_attention": (1e6, 1.0)}, window={"sliding_attention": 30},
                             sandwich=True)
        levels = [8, 7, 6, 6]
        self.levels = torch.tensor(levels)
        self.basis = torch.tensor([math.prod(levels[:i]) for i in range(4)])

    def fsq_latents(self, codes):
        """`[n_q, T]` codes -> `[T, n_q*4]` latents (`LiquidCodecQuantizer.decode` before `project_out`)."""
        idx = codes.T[..., None]  # [T, n_q, 1]
        half = self.levels // 2
        lat = ((idx // self.basis) % self.levels - half) / half
        return lat.reshape(lat.shape[0], -1).float()

    def decode(self, codes):
        sd = self.sd
        lat = linear(self.fsq_latents(codes), sd["quantizer.project_out.weight"], sd["quantizer.project_out.bias"])
        emb = linear(lat, sd["from_quantizer.weight"], sd["from_quantizer.bias"])  # [T, 512]
        up = F.interpolate(emb.T[None], emb.shape[0] * 6, mode="nearest-exact")[0].T
        h = self.decoder(up)
        spec = linear(h, sd["decoder.lin.weight"], sd["decoder.lin.bias"]).T  # [1282, F]
        logmag, angle = spec.chunk(2, 0)
        z = torch.polar(logmag.exp(), angle)
        n_fft, hop, win = 1280, 320, 1280
        window = torch.hann_window(win)
        frames = torch.fft.irfft(z, n_fft, dim=0) * window[:, None]  # [1280, F]
        nf = frames.shape[1]
        size = (nf - 1) * hop + win
        audio = torch.zeros(size)
        env = torch.zeros(size)
        for t in range(nf):
            audio[t * hop : t * hop + win] += frames[:, t]
            env[t * hop : t * hop + win] += window.square()
        trim = (win - hop) // 2
        return (audio[trim:-trim] / env[trim:-trim])


# ---------------------------------------------------------------- tokenizer


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


def asr_prompt(tok, n_audio, language="English"):
    head = encode(tok, f"<|im_start|>system\nPerform ASR in {language}.<|im_end|>\n<|im_start|>user\n")
    tail = encode(tok, "<|im_end|>\n<|im_start|>assistant\n")
    return [BOS] + head + [AUDIO_TOKEN] * n_audio + tail


def tts_prompt(tok, text, lang="American English", voice="female"):
    return [BOS] + encode(
        tok,
        f"<|im_start|>system\nPerform TTS in {lang} using a {voice} voice.<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<|audio_start|>",
    )


def round_list(x, nd=6):
    return [float(f"{v:.{nd}g}") for v in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/lm/lfm3-audio-2b")
    ap.add_argument("--wav", default="assets/harness/jfk_16k.wav")
    ap.add_argument("--out", default="crates/rlx-lfm3-audio/tests/fixtures/reference.json")
    ap.add_argument("--tts-text", default="Hello from Liquid.")
    ap.add_argument("--tts-frames", type=int, default=24)
    ap.add_argument("--asr-tokens", type=int, default=48)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    model = Lfm3Audio(args.model)
    codec = Codec(args.model)
    out = {"model": "LiquidAI/LFM3-Audio-2B", "wav": str(args.wav)}

    # ---- frontend
    samples = read_wav(args.wav)
    mel = log_mel(samples)
    out["mel"] = {
        "shape": list(mel.shape),
        "frame0": round_list(mel[:, 0].tolist()),
        "frame_mid": round_list(mel[:, mel.shape[1] // 2].tolist()),
        "sum": float(mel.double().sum()),
    }

    # ---- encoder + projector
    feats = model.audio_features(mel)
    out["audio_features"] = {
        "shape": list(feats.shape),
        "row0": round_list(feats[0, :64].tolist()),
        "row_last": round_list(feats[-1, :64].tolist()),
        "rms": float(feats.pow(2).mean().sqrt()),
    }

    # ---- one LM forward over a text prompt
    text_ids = [BOS] + encode(tok, "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n")
    h = model.hidden(model.embed[torch.tensor(text_ids)])
    logits = model.logits(h[-1])
    top = logits.topk(5)
    out["lm"] = {
        "ids": text_ids,
        "hidden_last": round_list(h[-1, :64].tolist()),
        "top5_ids": top.indices.tolist(),
        "top5_logits": round_list(top.values.tolist()),
    }

    # ---- greedy ASR (text-only: placeholders and mode markers suppressed)
    ids = asr_prompt(tok, feats.shape[0])
    emb = model.embed[torch.tensor(ids)].clone()
    emb[torch.tensor(ids) == AUDIO_TOKEN] = feats
    generated = []
    for _ in range(args.asr_tokens):
        hh = model.hidden(emb)
        lg = model.logits(hh[-1])
        lg[[AUDIO_TOKEN, AUDIO_OUT_TOKEN, MIXED_START, AUDIO_START]] = float("-inf")
        t = int(lg.argmax())
        generated.append(t)
        if t in EOS_IDS:
            break
        emb = torch.cat((emb, model.embed[t][None]))
    out["asr"] = {"prompt_ids": ids, "generated": generated, "text": tok.decode(generated)}

    # ---- depth decoder on fixed inputs
    g = torch.Generator().manual_seed(0)
    hid = torch.randn(2048, generator=g) * 0.5
    codes = torch.randint(0, AUDIO_EOS_CODE, (NUM_CODEBOOKS,), generator=g)
    masked = torch.tensor([i % 3 == 0 for i in range(NUM_CODEBOOKS)])
    dl = model.depth_logits(hid, codes, masked)
    out["depth"] = {
        "hidden": round_list(hid.tolist(), 9),
        "codes": codes.tolist(),
        "masked": masked.tolist(),
        "argmax": dl.argmax(-1).tolist(),
        "logits_head": [round_list(row[:32].tolist()) for row in dl],
    }

    # ---- greedy TTS frames
    ids = tts_prompt(tok, args.tts_text)
    emb = model.embed[torch.tensor(ids)]
    frames = []
    for _ in range(args.tts_frames):
        hh = model.hidden(emb)
        codes = model.greedy_frame(hh[-1])
        if bool((codes == AUDIO_EOS_CODE).any()):
            break
        frames.append(codes.tolist())
        emb = torch.cat((emb, model.frame_embed(codes)[None]))
    out["tts"] = {"text": args.tts_text, "prompt_ids": ids, "frames": frames}

    # ---- codec
    if frames:
        codes = torch.tensor(frames).T  # [12, T]
        audio = codec.decode(codes)
        out["codec"] = {
            "codes": frames,
            "num_samples": audio.numel(),
            "head": round_list(audio[:2048].tolist()),
            "rms": float(audio.pow(2).mean().sqrt()),
        }
        lat = codec.fsq_latents(torch.arange(16)[None].repeat(12, 1))
        out["fsq_first16"] = round_list(lat[:, :4].reshape(-1).tolist())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}: asr={out['asr']['text']!r}, tts frames={len(frames)}")


if __name__ == "__main__":
    main()
