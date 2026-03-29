//! Audio encoding for multiple output formats.

use std::fmt;

const SAMPLE_RATE: u32 = 24_000;

/// Supported audio output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

impl AudioFormat {
    /// Parse a format string (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "mp3" => Some(Self::Mp3),
            "opus" => Some(Self::Opus),
            "aac" => Some(Self::Aac),
            "flac" => Some(Self::Flac),
            "wav" => Some(Self::Wav),
            "pcm" => Some(Self::Pcm),
            _ => None,
        }
    }

    /// HTTP Content-Type header value.
    pub fn content_type(&self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Opus => "audio/ogg",
            Self::Aac => "audio/aac",
            Self::Flac => "audio/flac",
            Self::Wav => "audio/wav",
            Self::Pcm => "audio/pcm",
        }
    }
}

impl fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
            Self::Aac => "aac",
            Self::Flac => "flac",
            Self::Wav => "wav",
            Self::Pcm => "pcm",
        };
        write!(f, "{}", s)
    }
}

/// Encode audio samples in the specified format.
pub fn encode(samples: &[f32], format: AudioFormat) -> anyhow::Result<Vec<u8>> {
    match format {
        AudioFormat::Pcm => Ok(encode_pcm(samples)),
        AudioFormat::Wav => encode_wav(samples),
        AudioFormat::Mp3 => encode_mp3(samples),
        AudioFormat::Flac => encode_flac(samples),
        AudioFormat::Opus => encode_opus(samples),
        AudioFormat::Aac => encode_aac(samples),
    }
}

/// Encode f32 samples as raw 16-bit signed little-endian PCM.
pub fn encode_pcm(samples: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let i = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        buf.extend_from_slice(&i.to_le_bytes());
    }
    buf
}

/// Encode f32 samples as a WAV file (mono, 24 kHz, 16-bit PCM).
fn encode_wav(samples: &[f32]) -> anyhow::Result<Vec<u8>> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for &s in samples {
            let i = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(i)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

/// Encode f32 samples as MP3 (mono, 128 kbps CBR).
///
/// Resamples to 44100 Hz before encoding — 24 kHz is an MPEG-2 Layer III rate
/// that causes issues on some platforms and players.
fn encode_mp3(samples: &[f32]) -> anyhow::Result<Vec<u8>> {
    use mp3lame_encoder::{Builder, FlushNoGap, MonoPcm};

    let mp3_sample_rate = 44100u32;
    let samples_44k = resample(samples, SAMPLE_RATE, mp3_sample_rate);

    let mut encoder = Builder::new()
        .ok_or_else(|| anyhow::anyhow!("Failed to create MP3 encoder"))?;
    encoder
        .set_num_channels(1)
        .map_err(|e| anyhow::anyhow!("MP3 config error: {:?}", e))?;
    encoder
        .set_sample_rate(mp3_sample_rate)
        .map_err(|e| anyhow::anyhow!("MP3 config error: {:?}", e))?;
    encoder
        .set_brate(mp3lame_encoder::Bitrate::Kbps128)
        .map_err(|e| anyhow::anyhow!("MP3 config error: {:?}", e))?;
    encoder
        .set_quality(mp3lame_encoder::Quality::Best)
        .map_err(|e| anyhow::anyhow!("MP3 config error: {:?}", e))?;
    let mut encoder = encoder
        .build()
        .map_err(|e| anyhow::anyhow!("MP3 build error: {:?}", e))?;

    let pcm: Vec<i16> = samples_44k
        .iter()
        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
        .collect();

    let input = MonoPcm(&pcm);
    let mut mp3_out = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(pcm.len()));

    encoder
        .encode_to_vec(input, &mut mp3_out)
        .map_err(|e| anyhow::anyhow!("MP3 encode error: {:?}", e))?;
    encoder
        .flush_to_vec::<FlushNoGap>(&mut mp3_out)
        .map_err(|e| anyhow::anyhow!("MP3 flush error: {:?}", e))?;

    Ok(mp3_out)
}

/// Encode f32 samples as FLAC (mono, 24 kHz, 16-bit).
fn encode_flac(samples: &[f32]) -> anyhow::Result<Vec<u8>> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let pcm_i32: Vec<i32> = samples
        .iter()
        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i32)
        .collect();

    let source =
        flacenc::source::MemSource::from_samples(&pcm_i32, 1, 16, SAMPLE_RATE as usize);

    let config = flacenc::config::Encoder::default();
    let block_size = config.block_size;
    let config = config
        .into_verified()
        .map_err(|(_cfg, e)| anyhow::anyhow!("FLAC config error: {:?}", e))?;

    let flac_stream = flacenc::encode_with_fixed_block_size(&config, source, block_size)
        .map_err(|e| anyhow::anyhow!("FLAC encode error: {:?}", e))?;

    let mut sink = flacenc::bitsink::ByteSink::new();
    flac_stream
        .write(&mut sink)
        .map_err(|e| anyhow::anyhow!("FLAC write error: {:?}", e))?;

    Ok(sink.as_slice().to_vec())
}

/// Encode f32 samples as OGG Opus (48 kHz mono).
fn encode_opus(samples: &[f32]) -> anyhow::Result<Vec<u8>> {
    use audiopus::coder::Encoder as OpusEncoder;
    use audiopus::{Application, Channels, SampleRate};
    use ogg::writing::{PacketWriteEndInfo, PacketWriter};

    // Opus works best at 48 kHz
    let samples_48k = resample(samples, SAMPLE_RATE, 48000);

    // 20ms frame at 48 kHz = 960 samples
    let frame_size: usize = 960;

    let encoder = OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Voip)
        .map_err(|e| anyhow::anyhow!("Opus encoder init error: {:?}", e))?;

    let mut ogg_buf = Vec::new();
    let serial = 1u32;
    {
        let mut writer = PacketWriter::new(&mut ogg_buf);

        // OpusHead identification header (RFC 7845)
        let mut head = Vec::with_capacity(19);
        head.extend_from_slice(b"OpusHead");
        head.push(1); // version
        head.push(1); // channel count (mono)
        head.extend_from_slice(&0u16.to_le_bytes()); // pre-skip
        head.extend_from_slice(&48000u32.to_le_bytes()); // input sample rate
        head.extend_from_slice(&0i16.to_le_bytes()); // output gain
        head.push(0); // channel mapping family 0
        writer.write_packet(head, serial, PacketWriteEndInfo::EndPage, 0)?;

        // OpusTags comment header
        let vendor = b"kitten-tts-rs";
        let mut tags = Vec::new();
        tags.extend_from_slice(b"OpusTags");
        tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        tags.extend_from_slice(vendor);
        tags.extend_from_slice(&0u32.to_le_bytes());
        writer.write_packet(tags, serial, PacketWriteEndInfo::EndPage, 0)?;

        // Encode audio in 20ms frames
        let mut opus_out = vec![0u8; 4000];
        let mut granule: u64 = 0;
        let total_frames = (samples_48k.len() + frame_size - 1) / frame_size;

        for (i, chunk) in samples_48k.chunks(frame_size).enumerate() {
            let frame: Vec<f32> = if chunk.len() < frame_size {
                let mut padded = chunk.to_vec();
                padded.resize(frame_size, 0.0);
                padded
            } else {
                chunk.to_vec()
            };

            let encoded_len = encoder
                .encode_float(&frame, &mut opus_out)
                .map_err(|e| anyhow::anyhow!("Opus encode error: {:?}", e))?;

            granule += frame_size as u64;
            let end_info = if i == total_frames - 1 {
                PacketWriteEndInfo::EndStream
            } else {
                PacketWriteEndInfo::NormalPacket
            };

            writer.write_packet(
                opus_out[..encoded_len].to_vec(),
                serial,
                end_info,
                granule,
            )?;
        }
    }

    Ok(ogg_buf)
}

/// Resample audio from source_sr to target_sr using linear interpolation.
fn resample(samples: &[f32], source_sr: u32, target_sr: u32) -> Vec<f32> {
    if source_sr == target_sr || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = target_sr as f64 / source_sr as f64;
    let out_len = (samples.len() as f64 * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;

        let sample = if idx + 1 < samples.len() {
            samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
        } else {
            samples[samples.len() - 1]
        };
        output.push(sample);
    }

    output
}

/// AAC encoding is not currently supported.
fn encode_aac(_samples: &[f32]) -> anyhow::Result<Vec<u8>> {
    anyhow::bail!(
        "AAC encoding is not currently supported. \
         Supported formats: mp3, opus, flac, wav, pcm"
    )
}
