"""PRISM Metadata Audit Script.

Runs ShortsMetadataGenerator live against 9router Gemini 3.8 Flash
for at least 5 real sample clips from existing pipeline runs.
Collects and verifies Before/After comparisons without performing any uploads.
"""

import json
import logging
import sys
from pathlib import Path

from upload.metadata_generator import ShortsMetadataGenerator, is_raw_transcript

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prism_audit")

# Define the sample matrix from actual existing outputs
SAMPLES = [
    {
        "sample_id": "sample_01",
        "video_id": "vSBtdIBjUj0",
        "source_title": "NOVIA SITUMEANG IDOL DENGERIN CURHAT TENTANG TITIK TERENDAH DALAM HIDUP ANAK-ANAK SMA | #superyouth",
        "source_channel": "Ujian Nasional Superyouth",
        "clip_summary": "Curhat emosional anak SMA tentang beban dan ekspektasi berat menjadi anak pertama dalam keluarga",
        "clip_transcript": (
            "olehnya sebenarnya hal yang paling bikin akhir-akhir ini capek banget itu kayak kan aku anak pertama "
            "terus sudah mau kuliah gitu kan kayak harus jadi panutan nih buat adik gimana caranya kita bisa kayak "
            "jangan salah jurusan kampusnya bisa keterima atau enggak Terus bisaak dapat nilai yang bagus karenainc "
            "juga kan gitu ya Apa yang membuat kamu terus mikirin itu karena apa Karena mungkin sebagai anak pertama "
            "kamu banyak tuntutan atau seperti apa enggak secara langsung sih cuma kayak maksudnya orang tua pasti "
            "punya ekspektasi yang besar buat kita anak pertama"
        ),
        "old_title": "olehnya sebenarnya hal yang paling bikin akhir-akhir ini capek banget itu kayak #Shorts",
        "old_description": (
            "Auto Short from NOVIA SITUMEANG IDOL DENGERIN CURHAT TENTANG TITIK TERENDAH DALAM HIDUP ANAK-ANAK SMA | #superyouth\n\n"
            "#Shorts #Indonesia"
        ),
        "old_hashtags": ["#Shorts", "#Indonesia"],
    },
    {
        "sample_id": "sample_02",
        "video_id": "k2i-nxPTOqE",
        "source_title": "Pelajaran Hidup Yang Didapatkan Agnez Mo - Daniel Tetangga Kamu",
        "source_channel": "Daniel Mananta Network",
        "clip_summary": "Realisasi reflektif Agnez Mo saat menghadapi cobaan bertubi-tubi bahwa dirinya sendiri yang meminta dipersiapkan oleh Tuhan",
        "clip_transcript": (
            "Kau tahu, Gue sampai sempet bilang, ya ampun Tuhan iya sih mempersiapkan tapi jangan sampai segini banget. "
            "Itu dari yang gue punya masalah sama sahabat gue di sekolah, tIba-tiba gue ada masalah di karir gue. "
            "Tiba-tiba orang ngomongin gue, memfitnah gue segala macam sampai menjadi hal yang besar di televisi. "
            "Gue juga sama keluarga sempat ada perselisihan. Pada akhirnya gue ngerasa gue sangat sendirian. "
            "Terus gue ingat, oh iya ya gue yang minta untuk dipersiapkan."
        ),
        "old_title": "Kau tahu, Gue sampai sempet bilang, ya ampun Tuhan iya sih mempersiapkan tapi ja #Shorts",
        "old_description": (
            "Auto Short from Pelajaran Hidup Yang Didapatkan Agnez Mo - Daniel Tetangga Kamu\n\n"
            "#Shorts #Indonesia"
        ),
        "old_hashtags": ["#Shorts", "#Indonesia"],
    },
    {
        "sample_id": "sample_03",
        "video_id": "RGFMU19-DaA",
        "source_title": "Makna Kegagalan dan Penderitaan #DEEPTALKMALAM",
        "source_channel": "Hujan Tanda Tanya",
        "clip_summary": "Konsep psikologis bahwa rasa putus asa muncul saat penderitaan kehilangan makna, dibandingkan dengan perjuangan hidup atau mati yang punya tujuan",
        "clip_transcript": (
            "berusaha bikin rumus gitu ya dispeller Equals savouring minus minus jadi rasa putus asa itu timbul "
            "ketika kita tuh menderita tapi enggak ada maknanya Jadi kalau menderita tapi ada maknanya ada spiritnya "
            "kayak orang bertarung orang berjuang benar-benar supaya menang walaupun tersiksa gitu ya tapi dia masih "
            "masih merasakan spirit yang pakai orang zaman dulu Merdeka hidup atau mati itu bentar udah ada tujuannya "
            "mau tersiksa mau kalah dia tahu yang harus diperjuangkan gitu sehingga dia tidak menyerah"
        ),
        "old_title": "berusaha bikin rumus gitu ya dispeller Equals savouring minus minus jadi rasa pu #Shorts",
        "old_description": (
            "Auto Short from Makna Kegagalan dan Penderitaan #DEEPTALKMALAM\n\n"
            "#Shorts #Indonesia"
        ),
        "old_hashtags": ["#Shorts", "#Indonesia"],
    },
    {
        "sample_id": "sample_04",
        "video_id": "KdNHDwYYD2Y",
        "source_title": "Kasih aku 9 menit, aku bantu storytelling kamu lebih menarik sampai 159%!",
        "source_channel": "Nazhifuu",
        "clip_summary": "Teknik storytelling rehooking: cerita bukan seperti kora-kora tapi roller coaster penuh kejutan agar audiens tidak berhenti menonton",
        "clip_transcript": (
            "Cerita yang bagus itu bukan kayak kora-kora yang cuma naik turun satu kali doang, tapi kayak roller coaster "
            "yang ada banyak tanjakan dan juga turunan yang ngebuat orang jadi ngerasa nagih. Ini namanya rehooking. "
            "Setelah kita nyampaikan satu poin, satu pesan gitu ya, kita kasih pancingan baru lagi buat bikin audience "
            "kita jadi makin penasaran lagi. Misalkan contohnya gitu ya, setelah aku berhasil menyelesaikan misalkan "
            "masalah A gitu, aku kira permasalahan sudah selesai tapi ternyata ada permasalahan B yang jauh lebih gila lagi. "
            "Kata kunci yang bisa sering digunakan adalah kata-kata seperti tapi meskipun gitu ya."
        ),
        "old_title": "Cerita yang bagus itu bukan kayak kora-kora yang cuma naik turun satu kali doang",
        "old_description": (
            "Auto Short from Kasih aku 9 menit, aku bantu storytelling kamu lebih menarik sampai 159%!\n\n"
            "#Shorts #Indonesia"
        ),
        "old_hashtags": ["#Shorts", "#Indonesia"],
    },
    {
        "sample_id": "sample_05",
        "video_id": "y3XpKcnRm_w",
        "source_title": "VIRZA LOGIKA PERNAH KERJA BARENG LIMBAD!",
        "source_channel": "Padepokan Malam Kliwon",
        "clip_summary": "Cerita komedi pengalaman kerja bareng Master Limbad saat pandemi COVID: kena silent treatment dan cara Limbad nge-brief",
        "clip_transcript": (
            "kan enggak banyak ya. He. Tapi waktu itu tuh salah satu gua iya-ya aja buat nerima tuh COVID zaman COVID. "
            "Oh. Jadi ada kerjaan udah ambil aja dah. Limbat limbat dah. Heeh. Apa rasanya punya bos yang ngobrolnya dikit ya? "
            "Itu rasanya kayak kena silent treatment terus. Treatment limbat itu mah persona. Enggak aslinya mah ngomong "
            "nge-brief ng kayak apa? Limbat nge-brief. Dua kata lucu tuh. Limbat nge-brief ya gitu bahasa normal aja ya. "
            "Nanti kita syuting. H ya baru sampai situ doang dia bisa ngomong. sisanya. Hm."
        ),
        "old_title": "kan enggak banyak ya. He. Tapi waktu itu tuh salah satu gua iya-ya aja buat neri",
        "old_description": (
            "Auto Short from VIRZA LOGIKA PERNAH KERJA BARENG LIMBAD!\n\n"
            "#Shorts #Indonesia"
        ),
        "old_hashtags": ["#Shorts", "#Indonesia"],
    },
    {
        "sample_id": "sample_06",
        "video_id": "UxAHTdGR7do",
        "source_title": "Kalau Kamu Mau HIDUPMU Gini-Gini Terus, SKIP Aja Video Ini! | SUARA BERKELAS #172",
        "source_channel": "SUARA BERKELAS",
        "clip_summary": "Tiga pertanyaan krusial dalam mencintai dan mengenali diri sendiri: Who are you, What do you want, dan What you can give",
        "clip_transcript": (
            "He. Dan ketika kita berhasil, Mas Bilal, mencintai diri kita sendiri, di sini gongnya, Mas Bilal. "
            "Kita bisa mengenali diri sendiri dan menjawab tiga pertanyaan paling krusial di muka bumi ini, Mas. "
            "Pertama, who are you? Siapa kamu? H. Pertanyaan yang kedua, what do you want? Apa yang kamu mau? "
            "Dan pertanyaan yang paling lebih kita tidak mengerti, nomor tiga, what you can give? Apa yang bisa kamu berikan? "
            "Dan alhamdulillah saya berhasil menjawab pertanyaan itu, Mas Bilal. Dan di sinilah saya duduk bersama Mas Bilal."
        ),
        "old_title": "He. Dan ketika kita berhasil, Mas Bilal, mencintai diri kita sendiri, di sini go",
        "old_description": (
            "Auto Short from Kalau Kamu Mau HIDUPMU Gini-Gini Terus, SKIP Aja Video Ini! | SUARA BERKELAS #172\n\n"
            "#Shorts #Indonesia"
        ),
        "old_hashtags": ["#Shorts", "#Indonesia"],
    },
]


def main():
    gen = ShortsMetadataGenerator()
    results = []

    print("=" * 80)
    print("PRISM AUDIT: GENERATING SHORTS METADATA ACROSS 6 REAL SAMPLES (NO UPLOAD)")
    print("=" * 80)

    for idx, sample in enumerate(SAMPLES, 1):
        print(f"\n--- [SAMPLE {idx}/{len(SAMPLES)}] {sample['video_id']} ({sample['source_channel']}) ---")
        meta = gen.generate(
            clip_transcript=sample["clip_transcript"],
            source_title=sample["source_title"],
            source_channel=sample["source_channel"],
            clip_summary=sample["clip_summary"],
        )

        is_raw = is_raw_transcript(meta.selected_title, sample["clip_transcript"])
        title_len = len(meta.selected_title)

        print(f"Old Title:       {sample['old_title']}")
        print(f"Selected Title:  {meta.selected_title} ({title_len} chars)")
        print(f"Is Raw Copy?:    {is_raw} (Target: False)")
        print(f"Candidates (5):  {len(meta.title_candidates)}")
        for c_idx, cand in enumerate(meta.title_candidates, 1):
            star = " [*WINNER*]" if cand == meta.selected_title else ""
            print(f"  {c_idx}. {cand}{star}")
        print(f"Selection Reason: {meta.selection_reason}")
        print(f"Hashtags ({len(meta.hashtags)}): {meta.hashtags}")
        print(f"Quality Checks:   {meta.quality.model_dump()}")
        print(f"New Description:\n{meta.description}")

        # Invariant assertions
        assert not is_raw, f"Selected title was detected as raw transcript: {meta.selected_title}"
        assert title_len <= 85, f"Selected title exceeds 85 chars: {title_len}"
        assert len(meta.title_candidates) == 5, f"Expected 5 candidates, got {len(meta.title_candidates)}"
        assert 3 <= len(meta.hashtags) <= 6, f"Hashtags out of 3-6 range: {len(meta.hashtags)}"
        assert meta.hashtags[0] == "#Shorts", f"First hashtag must be #Shorts: {meta.hashtags[0]}"
        assert meta.quality.title_is_not_raw_transcript is True
        assert meta.quality.not_clickbait is True
        assert not meta.is_fallback, f"Sample {sample['video_id']} triggered fallback unexpectedly"

        results.append({
            "sample_id": sample["sample_id"],
            "video_id": sample["video_id"],
            "source_title": sample["source_title"],
            "source_channel": sample["source_channel"],
            "clip_summary": sample["clip_summary"],
            "clip_transcript": sample["clip_transcript"],
            "old_metadata": {
                "title": sample["old_title"],
                "description": sample["old_description"],
                "hashtags": sample["old_hashtags"],
            },
            "new_metadata": {
                "title_candidates": meta.title_candidates,
                "selected_title": meta.selected_title,
                "selection_reason": meta.selection_reason,
                "title_length": title_len,
                "description": meta.description,
                "hashtags": meta.hashtags,
                "quality": meta.quality.model_dump(),
                "is_fallback": meta.is_fallback,
            },
            "verification": {
                "is_raw_transcript": is_raw,
                "title_length_valid": title_len <= 85,
                "candidates_count_valid": len(meta.title_candidates) == 5,
                "hashtags_count_valid": 3 <= len(meta.hashtags) <= 6,
                "shorts_first_valid": meta.hashtags[0] == "#Shorts",
                "verdict": "VERIFIED_PASS",
            }
        })

    out_file = Path("output/prism_metadata_audit_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[OK] Saved all {len(results)} sample audit records to {out_file}")


if __name__ == "__main__":
    main()
