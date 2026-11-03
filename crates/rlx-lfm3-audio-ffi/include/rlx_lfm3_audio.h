// C ABI of rlx-lfm3-audio-ffi: LiquidAI LFM3-Audio-2B speech recognition and
// synthesis. Calls that fail return NULL and leave a message for
// lfm3a_last_error(). A model handle is not thread-safe.
#ifndef RLX_LFM3_AUDIO_H
#define RLX_LFM3_AUDIO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Lfm3aModel Lfm3aModel;

// Message of the last failed call on this thread ("" if none).
const char *lfm3a_last_error(void);

// Open the LM GGUF + mmproj GGUF. device: "metal" (NULL), "cpu", "coreml", ...
// low_memory: one LM graph, stage graphs dropped after use.
Lfm3aModel *lfm3a_open(const char *lm_path, const char *mmproj_path,
                       const char *device, bool low_memory);
void lfm3a_close(Lfm3aModel *m);

// Transcribe n mono samples at 16 kHz. language: English name ("English"),
// NULL lets the model pick. max_tokens 0 = 256. Free with lfm3a_free_string.
char *lfm3a_transcribe(Lfm3aModel *m, const float *pcm, size_t n,
                       const char *language, size_t max_tokens);

// Synthesize 24 kHz mono speech. language: code ("en"), voice: "female" /
// "male" (NULL = defaults). Free with lfm3a_free_audio(pcm, *out_len).
float *lfm3a_speak(Lfm3aModel *m, const char *text, const char *language,
                   const char *voice, uint64_t seed, size_t *out_len);

void lfm3a_free_string(char *s);
void lfm3a_free_audio(float *pcm, size_t len);

// Physical footprint of this process in bytes (iOS's memory limit).
uint64_t lfm3a_footprint(void);

#ifdef __cplusplus
}
#endif

#endif
