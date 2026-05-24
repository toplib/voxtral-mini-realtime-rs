#ifndef VOXTRAL_H
#define VOXTRAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ======================================================================
// ASR (Automatic Speech Recognition)
// ======================================================================

typedef struct VoxtralCtx VoxtralCtx;

// Create/destroy ASR context. Returns NULL on failure.
VoxtralCtx* voxtral_create(void);
void voxtral_destroy(VoxtralCtx* ctx);

// Load GGUF model and tokenizer JSON.
// Returns 0 on success, -1 on error (check voxtral_last_error).
int voxtral_load_model(VoxtralCtx* ctx, const char* gguf_path, const char* tokenizer_path);

// Transcribe a WAV file. On success, out_text is set to an allocated string
// that must be freed with voxtral_free_string. Returns 0 on success.
int voxtral_transcribe_file(VoxtralCtx* ctx, const char* wav_path, char** out_text);

// Transcribe raw PCM float32 audio. sample_rate is in Hz (will be resampled
// to 16kHz internally if needed). On success, out_text is set to an allocated
// string that must be freed with voxtral_free_string. Returns 0 on success.
int voxtral_transcribe_pcm(VoxtralCtx* ctx, const float* samples, int num_samples,
                           int sample_rate, char** out_text);

// Returns the last error message (valid until the next call on this context).
const char* voxtral_last_error(VoxtralCtx* ctx);

// Free a string allocated by the library.
void voxtral_free_string(char* s);

// ======================================================================
// TTS (Text-to-Speech)
// ======================================================================

typedef struct VoxtralTtsCtx VoxtralTtsCtx;

// Create/destroy TTS context. Returns NULL on failure.
VoxtralTtsCtx* voxtral_tts_create(void);
void voxtral_tts_destroy(VoxtralTtsCtx* ctx);

// Load TTS model from GGUF file. Returns 0 on success.
int voxtral_tts_load_model(VoxtralTtsCtx* ctx, const char* gguf_path);

// Load TTS tokenizer from tekken.json. Returns 0 on success.
int voxtral_tts_load_tokenizer(VoxtralTtsCtx* ctx, const char* json_path);

// Load a voice embedding from a SafeTensors file. Returns 0 on success.
int voxtral_tts_load_voice(VoxtralTtsCtx* ctx, const char* safetensors_path);

// Synthesize speech from text. On success, out_samples is allocated and must
// be freed with voxtral_tts_free_audio. out_sample_rate is always 24000.
// Returns 0 on success.
int voxtral_tts_speak(VoxtralTtsCtx* ctx, const char* text, float** out_samples,
                      int* out_num_samples, int* out_sample_rate);

// Free audio samples allocated by voxtral_tts_speak.
void voxtral_tts_free_audio(float* samples, int num_samples);

// Returns the last TTS error message (valid until the next call on this context).
const char* voxtral_tts_last_error(VoxtralTtsCtx* ctx);

#ifdef __cplusplus
}
#endif

#endif // VOXTRAL_H
