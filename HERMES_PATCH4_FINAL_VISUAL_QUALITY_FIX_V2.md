# Patch 4 — Final Visual Quality Fix Pass

**Do not merge/deploy/restart production. Do not start Mission 2.**

Independent reviewer telah memeriksa MP4 terbaru secara langsung.

## Yang sudah PASS

- Integer-second/keyframe flicker: tidak terlihat lagi.
- Duplicate subtitle Sample A: sudah hilang.
- Ghost subtitle pada blurred background: sudah hilang.
- ASS overlap Sample B sekitar ~2s: sudah tidak terlihat.
- Audio 48 kHz stereo: pertahankan.
- Indonesian-only invariant: pertahankan.

Jangan ubah bagian yang sudah lolos kecuali diperlukan untuk regression safety.

---

# 1. P0 — Existing Subtitle Harus Benar-Benar Readable di Mobile

Sample A sekarang mempertahankan subtitle asli secara utuh, tetapi layout `SUBTITLE_SAFE_FULL_WIDTH` mengecilkan keseluruhan source 16:9 menjadi foreground sempit di tengah kanvas 9:16.

Akibatnya:
- subtitle memang tidak terpotong,
- tetapi ukuran subtitle menjadi terlalu kecil untuk pengalaman short-form/mobile,
- wajah/subjek juga menjadi kecil,
- output kembali terasa seperti landscape video yang ditempel di tengah vertical canvas.

Ini belum memenuhi tujuan user:

> Jika source sudah punya subtitle, jangan tambahkan subtitle baru, tetapi pastikan subtitle hasil clip terlihat jelas.

## Requirement

Untuk burned-in subtitle, JANGAN generate/transcribe subtitle baru.

Gunakan subtitle asli sebagai source-of-truth dan pixel source asli.

Implementasikan layout `SUBTITLE_PRESERVE_COMPOSITE` atau equivalent:

1. Deteksi subtitle band/bbox dari source.
2. Area video DI ATAS subtitle band boleh di-reframe/crop lebih agresif untuk membuat wajah/subjek layak di portrait.
3. Subtitle band asli dipreservasi secara terpisah dari source frame.
4. Subtitle band asli boleh di-scale/reposition agar readable pada 1080x1920.
5. Jangan OCR lalu membuat kalimat subtitle baru jika tidak diperlukan.
6. Jangan tampilkan subtitle dua kali:
   - content/main layer harus mengecualikan subtitle source bila subtitle band ditampilkan terpisah.
7. Jangan menghasilkan ghost subtitle di background.
8. Subtitle source harus tetap identik dengan yang terlihat pada source.

Konsep:

```text
SOURCE 16:9
+--------------------------+
|                          |
|       FACE / VIDEO       |
|                          |
| ORIGINAL SUBTITLE BAND   |
+--------------------------+

              ↓

OUTPUT 9:16
+----------------+
|                |
|   PORTRAIT     |
|   SUBJECT      |
|                |
|----------------|
| ORIGINAL       |
| SUBTITLE BAND  |
| (readable)     |
+----------------+
```

Ini bukan "menambah subtitle baru".
Ini merekomposisi pixel subtitle source yang sudah ada.

## Acceptance

Pada Sample A:
- hanya 1 subtitle semantic layer,
- teks sama dengan source,
- tidak terpotong kiri/kanan,
- tidak terlalu kecil,
- terbaca nyaman pada preview mobile,
- wajah jauh lebih besar daripada layout full-width Patch 4 sebelumnya,
- tidak ada ghost text.

Berikan before/after screenshots pada timestamp:
- 5s
- 10s
- 20s
- 30s.

---

# 2. P0 — Generated Indonesian Subtitle Quality

Sample B tidak memiliki subtitle source sehingga Subtitle V2 memang benar untuk dibuat.

Namun reviewer menemukan subtitle generated yang tampak tidak masuk akal/gibberish pada bagian akhir, contoh sekitar ~35s:

```text
"ya kan dib kon tut bip mem tut"
```

Kalimat seperti itu tidak boleh lolos final quality gate sebagai subtitle Bahasa Indonesia.

## Requirement

Audit exact source audio + transcript pada interval tersebut.

Tentukan root cause:
- Whisper language tidak dipaksa `id`,
- ASR salah dengar,
- phrase fallback rusak,
- text normalization rusak,
- segmentation/chunking merusak token,
- atau source memang mengucapkan kata tersebut.

Jangan menebak.

### Jika ASR yang salah

Perbaiki secara konservatif:
- gunakan explicit Indonesian language hint (`id`) pada clip-local ASR,
- gunakan context/prompt Bahasa Indonesia bila engine mendukung,
- jangan mengubah kata hanya untuk terlihat "masuk akal" tanpa evidence audio,
- bila confidence terlalu rendah, jangan menghasilkan rangkaian token gibberish sebagai subtitle final.

Tambahkan subtitle quality guard sederhana:
- flag token sequence yang sangat rendah confidence / tidak masuk akal berdasarkan ASR confidence,
- gunakan fallback transcript source yang lebih terpercaya bila tersedia,
- atau tandai clip gagal subtitle-quality jika tidak bisa direkonstruksi secara terpercaya.

Jangan meminta Gemini mengarang dialog yang tidak terdengar.

## Acceptance

- Tidak ada obvious gibberish pada representative Sample B.
- Subtitle tetap sesuai audio.
- Bahasa Indonesia/slang natural tetap boleh.
- Word/phrase timing tetap sinkron.
- Tidak ada overlap subtitle.
- Tidak ada aggressive karaoke animation.

Berikan source audio/transcript evidence untuk interval yang diperbaiki.

---

# 3. Regression Gate

Semua yang sudah lolos harus tetap lolos:

- no keyframe flicker,
- no duplicate subtitle,
- no ghost subtitle,
- no subtitle overlap,
- scene-cut segmentation tetap aman,
- 48 kHz stereo,
- Indonesian-only gate,
- Gemini Visual Director fallback,
- QC production.

Render ulang:

### Sample A
Source dengan burned-in subtitle.
Expected:
- no generated subtitle,
- `SUBTITLE_PRESERVE_COMPOSITE`,
- original subtitle readable,
- subject framing lebih baik.

### Sample B
Source tanpa subtitle.
Expected:
- Subtitle V2 satu layer,
- no gibberish,
- no overlap,
- framing stabil.

Return:
- patched MP4 A/B,
- source/output comparison frames,
- subtitle decision result,
- ASR evidence untuk Sample B,
- filtergraph,
- ffprobe/QC,
- pytest output,
- final commit hash.

Status tetap:

`AWAITING INDEPENDENT VISUAL REVIEW`


---

# 4. P0 — Continuity / Watchability / Empty-Frame Prevention

Independent reviewer menemukan defect tambahan pada output aktual:

- beberapa bagian terasa seperti video "terpotong-potong",
- perpindahan framing/shot tidak smooth dan tidak nyaman ditonton,
- beberapa cut terjadi pada timing yang terasa salah,
- kadang hasil crop/shot tidak memiliki wajah, orang, atau objek utama sama sekali.

Ini adalah **hard product-quality issue**, bukan kosmetik.

## 4.1 Jangan Samakan "Scene Cut Valid" dengan "Edit Cut Bagus"

Scene-cut detector hanya menjawab:

> "Apakah source berganti shot?"

Itu TIDAK berarti setiap scene boundary harus menjadi edit/crop transition yang terlihat kasar.

Renderer harus menjaga kontinuitas visual di dalam shot dan hanya melakukan perubahan framing ketika ada alasan yang jelas.

Dilarang:
- crop jump tanpa alasan,
- camera position berubah drastis tiap keyframe,
- zoom/crop berpindah saat pembicara masih dalam shot yang sama,
- memotong visual di tengah gesture/wajah tanpa kebutuhan,
- berpindah ke frame kosong/background hanya karena detector kehilangan wajah sesaat.

## 4.2 Subject Presence Gate

Setiap candidate framing/crop harus memiliki `subject_presence` yang tervalidasi.

Minimal state:

```text
VALID_SUBJECT
TEMPORARY_FACE_LOSS
NO_VALID_SUBJECT
```

Primary subject dapat berupa:
- wajah/manusia,
- objek utama yang memang menjadi fokus konten,
- visual source yang secara semantic relevan menurut Visual Director.

Untuk podcast/talking-head/interview, default expectation:
`VALID_SUBJECT = human face/person`.

Jika candidate crop tidak memiliki subject:
- jangan gunakan crop tersebut,
- tahan framing terakhir yang masih valid untuk grace period singkat jika masih scene yang sama,
- atau gunakan deterministic safe fallback.

Jangan menampilkan crop kosong hanya karena face detector gagal satu sample frame.

## 4.3 Grace Period untuk Temporary Detection Loss

Face detector dapat gagal sesaat karena:
- kepala menoleh,
- motion blur,
- tangan menutup wajah,
- compression artifact,
- frame transisi.

Jangan langsung berpindah ke fallback setiap kali satu keyframe kehilangan face.

Gunakan temporal hysteresis/grace period.

Contoh semantics:

```text
valid face
valid face
miss 1 frame/keyframe
miss 2 frame/keyframe
valid face again
```

-> pertahankan subject track yang sama jika posisi/scene masih konsisten.

Baru anggap `NO_VALID_SUBJECT` jika loss berlangsung cukup lama atau scene benar-benar berubah.

Threshold harus deterministic dan configurable.

## 4.4 Framing Stability / Motion Limits

Tambahkan batas perubahan crop antar-keyframe dalam scene yang sama.

Contoh:
- max horizontal displacement per second,
- max vertical displacement per second,
- max zoom delta per second.

Jika perubahan detector melebihi batas tetapi scene tidak berubah:
- clamp/smooth,
- jangan hard jump.

Pada hard scene cut:
- instant reacquire diperbolehkan,
- tetapi setelah reacquire framing harus stabil.

## 4.5 Cut Timing / Semantic Continuity

Untuk generated editing events (punch-in, pacing cut, reframing event):

Jangan melakukan cut semata-mata berdasarkan timestamp fixed.

Gunakan gabungan:
- scene boundary,
- speech phrase boundary,
- silence/pause,
- transcript punctuation,
- Visual Director semantic signal.

Hindari cut:
- di tengah kata,
- di tengah kalimat tanpa alasan,
- saat gesture utama belum selesai,
- beberapa frame sebelum/after scene source yang sudah berubah secara alami.

Jika pacing engine masih disabled secara default, jangan mengaktifkannya hanya untuk menyelesaikan poin ini.

## 4.6 Visual Director Continuity Review

Gunakan Gemini Visual Director sebagai semantic validator, bukan sebagai renderer.

Tambahkan output terstruktur seperti:

```json
{
  "continuity_risk": "low|medium|high",
  "empty_subject_frames_detected": false,
  "recommended_hold_previous_framing": true,
  "reason": "Face detector briefly lost target but same speaker remains in scene."
}
```

Untuk representative clip, Visual Director boleh menandai:
- empty framing,
- subject lost,
- awkward camera jump,
- excessive crop movement,
- cut timing yang tidak natural.

Renderer tetap deterministic.

## 4.7 Automated / Perceptual Guardrails

Tambahkan test untuk:

- temporary one/two-keyframe face miss tidak menyebabkan crop kosong,
- no-valid-subject candidate ditolak,
- crop delta dalam scene tidak melebihi configured movement limit,
- hard scene cut melakukan reacquire tanpa interpolasi dari scene lama,
- no default/background-only crop ketika valid subject masih tersedia,
- punch-in/crop event tidak dibuat pada frame yang tidak memiliki subject.

Tambahkan representative frame audit minimal setiap 0.5–1.0 detik untuk dua sample.

Flag jika:
- tidak ada subject pada crop selama > configurable threshold,
- framing berubah terlalu jauh tanpa scene cut,
- subject keluar frame,
- wajah terpotong parah.

## 4.8 Acceptance Criteria

Patch belum PASS jika:
- video masih terasa choppy/terpotong-potong tanpa alasan,
- crop berpindah kasar di dalam shot yang sama,
- ada frame/interval kosong tanpa subject padahal source memiliki subject,
- face detector miss singkat langsung membuat framing jatuh ke background,
- cut timing terasa memotong speech/gesture secara jelas.

Representative Sample A dan B harus direview lagi secara visual setelah fix.

---

# 5. Updated Regression / Deliverables

Semua requirement sebelumnya tetap berlaku.

Tambahkan deliverables:

- subject-presence timeline,
- crop-position/zoom timeline,
- scene-cut timeline,
- daftar timestamp ketika detector kehilangan face,
- action yang dipilih saat face loss (`hold`, `reacquire`, `fallback`),
- VisualDirector continuity result,
- dense frame sheet untuk interval yang sebelumnya terasa choppy,
- before/after video comparison bila tersedia.

Status akhir tetap:

`AWAITING INDEPENDENT VISUAL REVIEW`
