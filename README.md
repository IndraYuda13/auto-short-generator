# Auto Short Generator (Nonstop Clipping & Auto-Upload Daemon)

Sistem otomatisasi backend nonstop yang memantau YouTube untuk podcast/talkshow terbaru, memilih video terbaik dengan AI (Gemini 3.8 Flash via 9router), mengekstrak transkrip kata demi kata, mendeteksi **viral hook** dengan konteks tuntas (durasi 30-55 detik), merender video format vertical 9:16 (1080x1920) dengan latar belakang blurred 16:9 dan dynamic karaoke subtitles gaya Alex Hormozi/MrBeast, serta mengunggahnya secara otomatis ke YouTube Shorts dan TikTok v2 API secara berulang tanpa henti.

Semua video dan klip dicatat dalam SQLite database (WAL mode) agar **tidak pernah memproses ulang video yang sama**.

---

## 📁 Struktur Project

```text
/root/projects/auto-short-generator/
├── config.py              # Konfigurasi Pydantic & load .env
├── db.py                  # Database SQLite WAL (deduplikasi & logging status)
├── llm_client.py          # HTTP Client ke 9router (Port 20128) OpenAI-compatible endpoint
├── searcher.py            # Pencari video (YouTube API v3 & smart yt-dlp fallback)
├── transcriber.py         # YouTube Transcript API & faster-whisper local INT8 fallback
├── analyzer.py            # Analisis golden hook viral oleh Gemini 3.8 Flash
├── renderer.py            # FFmpeg engine: crop 16:9 -> 9:16 blurred + karaoke ASS subtitles
├── uploader.py            # YouTube resumable OAuth v3 upload & TikTok Content Posting API v2
├── main.py                # Daemon runner nonstop loop & CLI flags
├── requirements.txt       # Dependencies Python
├── .env.example           # Template environment variables
├── .env                   # File konfigurasi aktif
├── data/                  # SQLite DB (app.db) & auth tokens
├── downloads/             # Temp directory download source video (otomatis dibersihkan)
└── output/                # Hasil render video 1080x1920 MP4 & subtitle .ass
```

---

## ⚙️ Persyaratan Sistem & Instalasi

### 1. Sistem
- Python 3.11+
- FFmpeg 6.0+ dengan dukungan `libx264` dan subtitle filter (`ass`)
- 9router aktif di port `127.0.0.1:20128` dengan model `gemini/gemini-3.8-flash`

### 2. Pasang Dependencies
```bash
cd /root/projects/auto-short-generator
pip install -r requirements.txt
```

---

## 🔑 Konfigurasi Environment (`.env`)

Salin template dan sesuaikan isinya:
```bash
cp .env.example .env
nano .env
```

Parameter utama:
- `ROUTER_BASE_URL`: URL 9router (default: `http://127.0.0.1:20128/v1`)
- `LLM_MODEL`: `gemini/gemini-3.8-flash`
- `SEARCH_QUERIES`: (Fallback) Kata kunci cadangan jika LLM offline. Pipeline default-nya otomatis memakai AI Gemini 3.8 Flash via 9router untuk men-generate topik dinamis dan general di setiap siklus.
- `COOKIES_FILE`: Path ke file cookies Netscape format untuk bypass bot check YouTube di datacenter VPS (default: `/root/projects/auto-short-generator/cookies.txt`).
- `TARGET_PLATFORMS`: `youtube,tiktok` atau `youtube` saja / `tiktok` saja / `local_only`
- `LOOP_INTERVAL_SECONDS`: Jeda antar siklus pengecekan (default: `300` = 5 menit)
- `MIN_CLIP_DURATION_SEC` / `MAX_CLIP_DURATION_SEC`: Rentang durasi klip (default: `30` s/d `55` detik)

---

## 🚀 Panduan Menjalankan

### 1. Test Koneksi & Database Stats
Cek statistik database deduplikasi kapan saja:
```bash
cd /root/projects/auto-short-generator
python3 main.py --stats
```

### 2. Jalankan 1 Siklus Percobaan (Dry-Run / Single Cycle)
Untuk menguji flow tanpa menunggu interval loop daemon:
```bash
python3 main.py --once
```

### 3. Jalankan Nonstop Daemon (Production Mode)

#### Opsi A: Menggunakan Background Process (`nohup`):
```bash
nohup python3 main.py > /root/projects/auto-short-generator/daemon.log 2>&1 &
```
Untuk memantau log secara real-time:
```bash
tail -f /root/projects/auto-short-generator/generator.log
```

#### Opsi B: Menggunakan Systemd Service (Recommended untuk Auto-Restart):
Buat service file `/etc/systemd/system/autoshort.service`:
```ini
[Unit]
Description=Auto Short Generator Daemon
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/projects/auto-short-generator
ExecStart=/usr/local/lib/hermes-agent/venv/bin/python3 /root/projects/auto-short-generator/main.py
Restart=always
RestartSec=10
EnvironmentFile=/root/projects/auto-short-generator/.env

[Install]
WantedBy=multi-user.target
```
Aktifkan dan jalankan:
```bash
systemctl daemon-reload
systemctl enable --now autoshort
systemctl status autoshort
```

---

## 📺 Konfigurasi Upload Platform

### 1. YouTube Data API v3 (OAuth 2.0 Resumable)
1. Buat project di Google Cloud Console, aktifkan **YouTube Data API v3**.
2. Buat kredensial **OAuth 2.0 Client ID** (Desktop Application).
3. Unduh file JSON kredensial dan simpan sebagai:
   `/root/projects/auto-short-generator/client_secrets.json`
4. Jalankan otentikasi satu kali:
   ```bash
   python3 main.py --auth-youtube
   ```
   Buka link otorisasi di browser, izinkan akses, dan token akan disimpan otomatis di `/root/projects/auto-short-generator/data/youtube_token.json` (auto-refresh selamanya).

### 2. TikTok Content Posting API v2 Resmi
1. Buat Developer App di [TikTok for Developers](https://developers.tiktok.com/).
2. Aktifkan permission `video.upload`.
3. Masukkan `TIKTOK_ACCESS_TOKEN` di `.env`.
4. *Catatan*: Jika token belum diisi, pipeline tetap berjalan merender klip dan menyimpannya di `/root/projects/auto-short-generator/output/` tanpa mengalami crash.

---

## 🛡️ Deduplikasi & Integritas Data (SQLite WAL)
- Database mencatat setiap `video_id` yang ditemukan.
- Sebelum mendownload, modul `searcher.py` mengecek tabel `videos`.
- Jika video berstatus `completed`, `processing`, atau `skipped`, video tersebut **dilewati seketika** sehingga tidak akan pernah diunduh atau dipotong 2 kali.
- File video sumber yang berukuran besar langsung dihapus dari disk setelah klip selesai dirender untuk menghemat kapasitas storage server.
