/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * C API of rlx-voice-link-ffi: BLE voice-link reassembly and int8
 * Conformer-CTC speech recognition. Handles are not thread-safe.
 */
#ifndef RLX_VOICE_LINK_H
#define RLX_VOICE_LINK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RvlReceiver RvlReceiver;
typedef struct RvlAsr RvlAsr;

/* Receiver keeping utterances up to max_samples (16 kHz). */
RvlReceiver *rvl_receiver_new(size_t max_samples);
void rvl_receiver_free(RvlReceiver *r);
/* 1 = utterance complete, 0 = audio appended, -1 = dropped / bad argument. */
int rvl_receiver_push(RvlReceiver *r, const uint8_t *data, size_t len);
/* Last utterance; *out valid until the next push. Returns the sample count. */
size_t rvl_receiver_utterance(const RvlReceiver *r, const int16_t **out);
uint32_t rvl_receiver_lost(const RvlReceiver *r);

/* Load an .rcc model from memory (copied). NULL if invalid. */
RvlAsr *rvl_asr_open(const uint8_t *data, size_t len);
void rvl_asr_free(RvlAsr *a);
/* 16 kHz PCM -> UTF-8 (not NUL-terminated). Bytes written, or < 0 on error. */
ptrdiff_t rvl_asr_transcribe(RvlAsr *a, const int16_t *pcm, size_t n, uint8_t *out, size_t cap);

#ifdef __cplusplus
}
#endif
#endif
