use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::AssertUnwindSafe;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;

use crate::audio::io::load_wav;
use crate::audio::mel::{MelConfig, MelSpectrogram};
use crate::audio::pad::{pad_audio, PadConfig};
use crate::audio::resample::resample_to_16k;
use crate::audio::AudioBuffer;
use crate::gguf::loader::Q4ModelLoader;
use crate::gguf::model::Q4VoxtralModel;
use crate::models::time_embedding::TimeEmbedding;
use crate::tokenizer::VoxtralTokenizer;

type Backend = Wgpu<f32, i32>;

// -----------------------------------------------------------------------
// VoxtralCtx — ASR context
// -----------------------------------------------------------------------

#[repr(C)]
pub struct VoxtralCtx {
    model: Option<Q4VoxtralModel>,
    tokenizer: Option<VoxtralTokenizer>,
    mel_extractor: MelSpectrogram,
    pad_config: PadConfig,
    time_embed: TimeEmbedding,
    device: WgpuDevice,
    last_error: Option<CString>,
}

impl VoxtralCtx {
    fn set_error(&mut self, msg: impl Into<String>) {
        self.last_error = CString::new(msg.into()).ok();
    }

    fn audio_to_mel(&self, audio: &AudioBuffer) -> Result<Tensor<Backend, 3>, String> {
        let padded = pad_audio(audio, &self.pad_config);
        let mel = self.mel_extractor.compute_log(&padded.samples);
        let n_frames = mel.len();
        if n_frames == 0 {
            return Err("Audio too short to produce mel frames".to_string());
        }
        let n_mels = mel[0].len();

        let mut mel_transposed = vec![vec![0.0f32; n_frames]; n_mels];
        for (frame_idx, frame) in mel.iter().enumerate() {
            for (mel_idx, &val) in frame.iter().enumerate() {
                mel_transposed[mel_idx][frame_idx] = val;
            }
        }
        let mel_flat: Vec<f32> = mel_transposed.into_iter().flatten().collect();
        Ok(Tensor::from_data(
            burn::tensor::TensorData::new(mel_flat, [1, n_mels, n_frames]),
            &self.device,
        ))
    }

    fn transcribe_impl(&self, audio: &AudioBuffer) -> Result<String, String> {
        let model = self
            .model
            .as_ref()
            .ok_or("Model not loaded. Call voxtral_load_model first.")?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or("Tokenizer not loaded.")?;

        let mut buf = audio.clone();
        buf.peak_normalize(0.95);

        let mel_tensor = self.audio_to_mel(&buf)?;
        let t_embed = self.time_embed.embed::<Backend>(6.0, &self.device);
        let tokens = model.transcribe_streaming(mel_tensor, t_embed);

        let text_tokens: Vec<u32> = tokens
            .iter()
            .filter(|&&t| t >= 1000)
            .map(|&t| t as u32)
            .collect();

        tokenizer
            .decode(&text_tokens)
            .map_err(|e| format!("Failed to decode tokens: {}", e))
    }
}

fn set_ctx_error(ctx: &mut VoxtralCtx, e: Box<dyn std::any::Any + Send>) {
    let msg = if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "Unknown panic in voxtral library".to_string()
    };
    ctx.set_error(msg);
}

fn set_tts_error(ctx: &mut VoxtralTtsCtx, e: Box<dyn std::any::Any + Send>) {
    let msg = if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "Unknown panic in voxtral TTS library".to_string()
    };
    ctx.set_error(msg);
}

// ======================================================================
// ASR API
// ======================================================================

#[no_mangle]
pub unsafe extern "C" fn voxtral_create() -> *mut VoxtralCtx {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        Box::into_raw(Box::new(VoxtralCtx {
            model: None,
            tokenizer: None,
            mel_extractor: MelSpectrogram::new(MelConfig::voxtral()),
            pad_config: PadConfig::voxtral(),
            time_embed: TimeEmbedding::new(3072),
            device: WgpuDevice::default(),
            last_error: None,
        }))
    }));
    match result {
        Ok(ptr) => ptr,
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn voxtral_destroy(ctx: *mut VoxtralCtx) {
    if ctx.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(Box::from_raw(ctx));
    }));
}

#[no_mangle]
pub unsafe extern "C" fn voxtral_load_model(
    ctx: *mut VoxtralCtx,
    gguf_path: *const c_char,
    tokenizer_path: *const c_char,
) -> i32 {
    let ctx = match (unsafe { ctx.as_mut() }, unsafe { gguf_path.as_ref() }, unsafe {
        tokenizer_path.as_ref()
    }) {
        (Some(c), Some(_), Some(_)) => c,
        _ => return -1,
    };

    let gguf = unsafe { CStr::from_ptr(gguf_path) }
        .to_string_lossy()
        .into_owned();
    let tokenizer = unsafe { CStr::from_ptr(tokenizer_path) }
        .to_string_lossy()
        .into_owned();

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let tok = VoxtralTokenizer::from_file(&tokenizer)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        ctx.tokenizer = Some(tok);

        let mut loader = Q4ModelLoader::from_file(std::path::Path::new(&gguf))
            .map_err(|e| format!("Failed to parse GGUF: {}", e))?;
        ctx.model = Some(
            loader
                .load(&ctx.device)
                .map_err(|e| format!("Failed to load Q4 model: {}", e))?,
        );
        Ok(())
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            ctx.set_error(e);
            -1
        }
        Err(e) => {
            set_ctx_error(ctx, e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn voxtral_transcribe_file(
    ctx: *mut VoxtralCtx,
    wav_path: *const c_char,
    out_text: *mut *mut c_char,
) -> i32 {
    let ctx = match unsafe { ctx.as_mut() } {
        Some(c) => c,
        None => return -1,
    };
    let path = match unsafe { wav_path.as_ref() } {
        Some(_) => unsafe { CStr::from_ptr(wav_path) }
            .to_string_lossy()
            .into_owned(),
        None => {
            ctx.set_error("wav_path is null");
            return -1;
        }
    };

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<String, String> {
        let mut audio =
            load_wav(&path).map_err(|e| format!("Failed to load WAV file: {}", e))?;
        if audio.sample_rate != 16000 {
            audio = resample_to_16k(&audio)
                .map_err(|e| format!("Failed to resample audio: {}", e))?;
        }
        ctx.transcribe_impl(&audio)
    }));

    match result {
        Ok(Ok(text)) => match CString::new(text) {
            Ok(cs) => {
                unsafe { *out_text = cs.into_raw() };
                0
            }
            Err(_) => {
                ctx.set_error("Transcription result contains null byte");
                -1
            }
        },
        Ok(Err(e)) => {
            ctx.set_error(e);
            -1
        }
        Err(e) => {
            set_ctx_error(ctx, e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn voxtral_transcribe_pcm(
    ctx: *mut VoxtralCtx,
    samples: *const f32,
    num_samples: i32,
    sample_rate: i32,
    out_text: *mut *mut c_char,
) -> i32 {
    let ctx = match unsafe { ctx.as_mut() } {
        Some(c) => c,
        None => return -1,
    };
    if samples.is_null() || num_samples <= 0 || sample_rate <= 0 {
        ctx.set_error("Invalid PCM parameters");
        return -1;
    }

    let rate = sample_rate as u32;
    let num = num_samples as usize;
    let slice = unsafe { std::slice::from_raw_parts(samples, num) };

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<String, String> {
        let mut audio = AudioBuffer::new(slice.to_vec(), rate);
        if audio.sample_rate != 16000 {
            audio = resample_to_16k(&audio)
                .map_err(|e| format!("Failed to resample audio: {}", e))?;
        }
        ctx.transcribe_impl(&audio)
    }));

    match result {
        Ok(Ok(text)) => match CString::new(text) {
            Ok(cs) => {
                unsafe { *out_text = cs.into_raw() };
                0
            }
            Err(_) => {
                ctx.set_error("Transcription result contains null byte");
                -1
            }
        },
        Ok(Err(e)) => {
            ctx.set_error(e);
            -1
        }
        Err(e) => {
            set_ctx_error(ctx, e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn voxtral_last_error(ctx: *const VoxtralCtx) -> *const c_char {
    match unsafe { ctx.as_ref() } {
        Some(c) => c
            .last_error
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null()),
        None => std::ptr::null(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn voxtral_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = CString::from_raw(s);
    }));
}

// ======================================================================
// TTS API
// ======================================================================

use crate::gguf::tts_loader::Q4TtsModelLoader;
use crate::gguf::tts_model::{Q4FmTransformer, Q4TtsBackbone};
use crate::tts::codec::CodecDecoder;
use crate::tts::config::AudioCodebookLayout;
use crate::tts::embeddings::AudioCodebookEmbeddings;
use crate::tts::voice::load_voice_from_bytes;
use crate::tokenizer::TekkenEncoder;

#[repr(C)]
pub struct VoxtralTtsCtx {
    backbone: Option<Q4TtsBackbone>,
    fm: Option<Q4FmTransformer>,
    codec: Option<CodecDecoder<Wgpu>>,
    codebook: Option<AudioCodebookEmbeddings<Wgpu>>,
    device: WgpuDevice,
    voice_embed: Option<Tensor<Wgpu, 2>>,
    tokenizer: Option<TekkenEncoder>,
    last_error: Option<CString>,
}

impl VoxtralTtsCtx {
    fn set_error(&mut self, msg: impl Into<String>) {
        self.last_error = CString::new(msg.into()).ok();
    }
}

#[no_mangle]
pub unsafe extern "C" fn voxtral_tts_create() -> *mut VoxtralTtsCtx {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        Box::into_raw(Box::new(VoxtralTtsCtx {
            backbone: None,
            fm: None,
            codec: None,
            codebook: None,
            device: WgpuDevice::default(),
            voice_embed: None,
            tokenizer: None,
            last_error: None,
        }))
    }));
    match result {
        Ok(ptr) => ptr,
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn voxtral_tts_destroy(ctx: *mut VoxtralTtsCtx) {
    if ctx.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(Box::from_raw(ctx));
    }));
}

#[no_mangle]
pub unsafe extern "C" fn voxtral_tts_load_model(
    ctx: *mut VoxtralTtsCtx,
    gguf_path: *const c_char,
) -> i32 {
    let ctx = match (unsafe { ctx.as_mut() }, unsafe { gguf_path.as_ref() }) {
        (Some(c), Some(_)) => c,
        _ => return -1,
    };
    let path = unsafe { CStr::from_ptr(gguf_path) }
        .to_string_lossy()
        .into_owned();

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let mut loader = Q4TtsModelLoader::from_file(std::path::Path::new(&path))
            .map_err(|e| format!("Failed to parse TTS GGUF: {}", e))?;
        let (backbone, fm, codec) = loader
            .load(&ctx.device)
            .map_err(|e| format!("Failed to load TTS model: {}", e))?;

        let layout = AudioCodebookLayout::default();
        let codebook =
            AudioCodebookEmbeddings::new(backbone.audio_codebook_embeddings().clone(), layout);

        ctx.backbone = Some(backbone);
        ctx.fm = Some(fm);
        ctx.codec = Some(codec);
        ctx.codebook = Some(codebook);
        Ok(())
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            ctx.set_error(e);
            -1
        }
        Err(e) => {
            set_tts_error(ctx, e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn voxtral_tts_load_tokenizer(
    ctx: *mut VoxtralTtsCtx,
    json_path: *const c_char,
) -> i32 {
    let ctx = match (unsafe { ctx.as_mut() }, unsafe { json_path.as_ref() }) {
        (Some(c), Some(_)) => c,
        _ => return -1,
    };
    let path = unsafe { CStr::from_ptr(json_path) }
        .to_string_lossy()
        .into_owned();

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let encoder = TekkenEncoder::from_file(std::path::Path::new(&path))
            .map_err(|e| format!("Failed to load TTS tokenizer: {}", e))?;
        ctx.tokenizer = Some(encoder);
        Ok(())
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            ctx.set_error(e);
            -1
        }
        Err(e) => {
            set_tts_error(ctx, e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn voxtral_tts_load_voice(
    ctx: *mut VoxtralTtsCtx,
    safetensors_path: *const c_char,
) -> i32 {
    let ctx = match (unsafe { ctx.as_mut() }, unsafe { safetensors_path.as_ref() }) {
        (Some(c), Some(_)) => c,
        _ => return -1,
    };
    let path = unsafe { CStr::from_ptr(safetensors_path) }
        .to_string_lossy()
        .into_owned();

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let bytes =
            std::fs::read(&path).map_err(|e| format!("Failed to read voice file: {}", e))?;
        let embed: Tensor<Wgpu, 2> =
            load_voice_from_bytes(&bytes, 3072, &ctx.device)
                .map_err(|e| format!("Failed to load voice: {}", e))?;
        ctx.voice_embed = Some(embed);
        Ok(())
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            ctx.set_error(e);
            -1
        }
        Err(e) => {
            set_tts_error(ctx, e);
            -1
        }
    }
}

struct TtsOutput {
    samples: Vec<f32>,
    sample_rate: i32,
}

#[no_mangle]
pub unsafe extern "C" fn voxtral_tts_speak(
    ctx: *mut VoxtralTtsCtx,
    text: *const c_char,
    out_samples: *mut *mut f32,
    out_num_samples: *mut i32,
    out_sample_rate: *mut i32,
) -> i32 {
    let ctx = match unsafe { ctx.as_mut() } {
        Some(c) => c,
        None => return -1,
    };
    let input = match unsafe { text.as_ref() } {
        Some(_) => unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned(),
        None => {
            ctx.set_error("text is null");
            return -1;
        }
    };

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<TtsOutput, String> {
        let backbone = ctx
            .backbone
            .as_ref()
            .ok_or("TTS model not loaded. Call voxtral_tts_load_model first.")?;
        let fm = ctx.fm.as_ref().ok_or("FM transformer not loaded.")?;
        let codec = ctx.codec.as_ref().ok_or("Codec not loaded.")?;
        let codebook = ctx.codebook.as_ref().ok_or("Codebook not loaded.")?;
        let voice_embed = ctx
            .voice_embed
            .as_ref()
            .ok_or("No voice loaded. Call voxtral_tts_load_voice first.")?;
        let tokenizer = ctx
            .tokenizer
            .as_ref()
            .ok_or("Tokenizer not loaded. Call voxtral_tts_load_tokenizer first.")?;

        let text_ids = tokenizer.encode(&input);

        let special = crate::tts::config::TtsSpecialTokens::default();
        let dim = backbone.d_model();

        let bos_embed = backbone.embed_tokens_from_ids(&[special.bos_token_id as i32], 1, 1);
        let begin_audio_embed =
            backbone.embed_tokens_from_ids(&[special.begin_audio_token_id as i32], 1, 1);
        let next_audio_text_embed =
            backbone.embed_tokens_from_ids(&[special.next_audio_text_token_id as i32], 1, 1);
        let repeat_audio_text_embed =
            backbone.embed_tokens_from_ids(&[special.repeat_audio_text_token_id as i32], 1, 1);

        let text_ids_i32: Vec<i32> = text_ids.iter().map(|&id| id as i32).collect();
        let text_embeds = if text_ids_i32.is_empty() {
            Tensor::<Wgpu, 3>::zeros([1, 0, dim], &ctx.device)
        } else {
            backbone.embed_tokens_from_ids(&text_ids_i32, 1, text_ids_i32.len())
        };

        let voice_3d = voice_embed.clone().unsqueeze_dim::<3>(0);

        let input_sequence = Tensor::cat(
            vec![
                bos_embed,
                begin_audio_embed.clone(),
                voice_3d,
                next_audio_text_embed,
                text_embeds,
                repeat_audio_text_embed,
                begin_audio_embed,
            ],
            1,
        );

        let max_frames = 750;
        let frames = pollster::block_on(
            backbone.generate_async(input_sequence, fm, codebook, max_frames),
        )
        .map_err(|e| format!("TTS generation failed: {}", e))?;

        if frames.is_empty() {
            return Ok(TtsOutput {
                samples: Vec::new(),
                sample_rate: 24000,
            });
        }

        let n_frames = frames.len();
        let semantic_indices: Vec<usize> = frames.iter().map(|f| f.semantic_idx).collect();

        let mut acoustic_data = Vec::with_capacity(n_frames * 36);
        for frame in &frames {
            for &level in &frame.acoustic_levels {
                acoustic_data.push(level as f32);
            }
        }
        let acoustic_indices: Tensor<Wgpu, 2> = Tensor::from_data(
            burn::tensor::TensorData::new(acoustic_data, [n_frames, 36]),
            &ctx.device,
        );

        let waveform = codec.decode(&semantic_indices, acoustic_indices);
        let [_batch, total_samples] = waveform.dims();
        let mut samples: Vec<f32> = waveform.into_data().to_vec::<f32>().unwrap_or_default();
        samples.truncate(total_samples);

        let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if peak > 1e-6 {
            let gain = 0.95 / peak;
            for s in &mut samples {
                *s *= gain;
            }
        }

        Ok(TtsOutput {
            samples,
            sample_rate: 24000,
        })
    }));

    match result {
        Ok(Ok(output)) => {
            let num = output.samples.len() as i32;
            if num == 0 {
                unsafe {
                    *out_samples = std::ptr::null_mut();
                    *out_num_samples = 0;
                    *out_sample_rate = output.sample_rate;
                }
                return 0;
            }
            let layout = std::alloc::Layout::array::<f32>(num as usize).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) } as *mut f32;
            if ptr.is_null() {
                ctx.set_error("Failed to allocate audio output buffer");
                return -1;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(output.samples.as_ptr(), ptr, num as usize);
                *out_samples = ptr;
                *out_num_samples = num;
                *out_sample_rate = output.sample_rate;
            }
            0
        }
        Ok(Err(e)) => {
            ctx.set_error(e);
            -1
        }
        Err(e) => {
            set_tts_error(ctx, e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn voxtral_tts_free_audio(samples: *mut f32, num_samples: i32) {
    if samples.is_null() || num_samples <= 0 {
        return;
    }
    let layout = std::alloc::Layout::array::<f32>(num_samples as usize).unwrap();
    unsafe {
        std::alloc::dealloc(samples as *mut u8, layout);
    }
}

#[no_mangle]
pub unsafe extern "C" fn voxtral_tts_last_error(ctx: *const VoxtralTtsCtx) -> *const c_char {
    match unsafe { ctx.as_ref() } {
        Some(c) => c
            .last_error
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null()),
        None => std::ptr::null(),
    }
}
