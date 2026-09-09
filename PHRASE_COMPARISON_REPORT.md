# Auto Clipper V3.1 — Critical Phrase Comparison Report

**Artifacts Tested & Generated:**
- Clip 1: `/root/projects/auto-short-generator-v3/output/fQbpsIQpi08_308_350_v3.mp4` (ASS: `/root/projects/auto-short-generator-v3/output/fQbpsIQpi08_308_350_v3.ass`)
- Clip 2: `/root/projects/auto-short-generator-v3/output/UxAHTdGR7do_402_443_v3.mp4` (ASS: `/root/projects/auto-short-generator-v3/output/UxAHTdGR7do_402_443_v3.ass`)

---

## 1. Critical Phrase Comparison Table

| Target Term | Clip ID | Old ASR Text (Base ASR) | Source Transcript (YouTube Caption) | Gemini Verifier (9router Video) | Final Fused Text (V3.1 ASS) | Audio Span (Whisper Word Timestamps) | Confidence | Evidence Tags |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **muka bumi** | `UxAHTdGR7do_402_443` | `crucial di mukabomi ini mas.` | `tiga pertanyaan paling krusial di muka bumi ini, Mas.` | `krusial di muka bumi ini, Mas.` | `krusial di muka bumi ini,` | `10.52s - 12.30s` | **HIGH** | `whisper, source_caption, gemini_native_video` |
| **pertama** | `UxAHTdGR7do_402_443` | `Peren pertama,` | `Pertama, who are you? Siapa kamu?` | `Pertama,` | `Mas. Pertama,` | `12.32s - 14.24s` | **HIGH** | `whisper, source_caption, gemini_native_video` |
| **who are you? siapa kamu?** | `UxAHTdGR7do_402_443` | `huayu siapa kamu?` | `who are you? Siapa kamu?` | `who are you? Siapa kamu?` | `who are you?` / `Siapa kamu?` | `14.46s - 16.84s` | **HIGH** | `whisper, source_caption, gemini_native_video` |
| **nomor tiga** | `UxAHTdGR7do_402_443` | `Nama tiga,` | `nomor tiga, what you can give?` | `nomor tiga,` | `nomor tiga,` | `26.63s - 27.55s` | **HIGH** | `whisper, source_caption, gemini_native_video` |
| **senyumnya sumringah** | `fQbpsIQpi08_308_350` | `sunyumnya menyosum meringah` | `senyumnya sumur ringah kayak gitu ya` | `senyumnya sumringah kayak gitu ya` | `senyumnya sumringah,` | `21.59s - 23.11s` | **HIGH** | `whisper, source_caption, gemini_native_video` |
| **satpam** | `fQbpsIQpi08_308_350` | `Untung sapangnya baik` | `Untung spamnya bayik enggak bayar ini parkir` | `Untung satpamnya baik, enggak bayar ini parkir` | `Wow. Untung satpamnya baik,` | `12.76s - 14.04s` | **HIGH** | `whisper, source_caption, gemini_native_video` |
| **aku S1** | `fQbpsIQpi08_308_350` | `Aku yang satu` | `Aku W tuh aku enggak goblok gitu loh` | `Aku S1 tuh, aku enggak goblok gitu loh` | `aku S1,` | `29.59s - 31.15s` | **HIGH** | `whisper, source_caption, gemini_native_video` |

---

## 2. Gemini Native Video QC Results (via 9router)

### Clip 1: `fQbpsIQpi08_308_350_v3.mp4`
- **Passed**: `True` (Score: 72/100)
- **subtitle_timing**: `PASS`
- **subtitle_overlap**: `PASS` (0 overlapping events)
- **subtitle_linger**: `PASS`
- **subtitle_text_accuracy**: `PASS`
- **obvious_transcription_errors**: `[]`
- **blocking_reasons**: `[]`
- **Gemini Director Summary**: *"Subtitle tersinkronisasi presisi dan akurat secara verbatim. Visual dan framing tertata baik tanpa tumpang tindih teks."*

### Clip 2: `UxAHTdGR7do_402_443_v3.mp4`
- **Passed**: `True` (Score: 72/100)
- **subtitle_timing**: `PASS`
- **subtitle_overlap**: `PASS` (0 overlapping events)
- **subtitle_linger**: `PASS`
- **subtitle_text_accuracy**: `PASS`
- **obvious_transcription_errors**: `[]`
- **blocking_reasons**: `[]`
- **Gemini Director Summary**: *"Sinkronisasi subtitle presisi, teks verbatim akurat tanpa overlap, dan framing visual aman pada safe-zone vertikal."*

---

## 3. Subtitle Timeline Invariant Audit
- Clip 1: 39 events, 0 invalid intervals, 0 overlapping events, max simultaneous = 1.
- Clip 2: 25 events, 0 invalid intervals, 0 overlapping events, max simultaneous = 1.
- Clear during speech silence gaps (>300ms) enforced.
- Style: Montserrat 46, vertical margin 440, horizontal margin 90, automatic `\N` line-break for phrases >3 words.
