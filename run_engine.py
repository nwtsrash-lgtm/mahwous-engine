"""
run_engine.py — سكربت التشغيل الآلي لمحرك مهووس v10.0
=======================================================
يعمل في بيئة GitHub Actions بشكل تلقائي كامل.

المسارات:
  input/store/        ← ملف(ات) CSV متجر مهووس
  input/competitors/  ← ملفات CSV المنافسين
  input/brands/       ← قائمة الماركات (اختياري)
  output/             ← النتائج (يُنشأ تلقائياً)
"""

from __future__ import annotations
import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# ── إعداد السجل ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mahwous-runner")

# ── المسارات ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
INPUT_STORE   = BASE_DIR / "input" / "store"
INPUT_COMP    = BASE_DIR / "input" / "competitors"
INPUT_BRANDS  = BASE_DIR / "input" / "brands"
OUTPUT_DIR    = BASE_DIR / "output"
CHECKPOINT    = OUTPUT_DIR / ".checkpoint.json"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── متغيرات البيئة ────────────────────────────────────────────────────────────
ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
USE_LLM        = os.environ.get("USE_LLM", "true").lower() == "true"
DESCRIBE_ONLY  = os.environ.get("DESCRIBE_ONLY", "false").lower() == "true"
MAX_PRODUCTS   = int(os.environ.get("MAX_PRODUCTS", "0"))   # 0 = بلا حد
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", "500"))

# ── الاستيرادات ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR))
from logic import (
    MahwousEngine, SemanticIndex,
    export_salla_csv, export_detailed_csv, export_brands_csv,
    load_brands, load_competitor_products, load_store_products,
)
from describe import generate_batch


# ─── مساعدات ─────────────────────────────────────────────────────────────────

def _load_csv_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.csv"))
    if not files:
        log.warning(f"⚠️ لا توجد ملفات CSV في: {folder}")
    else:
        log.info(f"📂 {len(files)} ملف في {folder.name}/: {[f.name for f in files]}")
    return files


def _progress_cb(i: int, total: int, name: str) -> None:
    if i % 200 == 0 or i <= 5 or i == total:
        pct = i / max(total, 1) * 100
        log.info(f"  ⚙️  [{pct:5.1f}%] {i:,}/{total:,} — {name[:55]}")


def _desc_progress_cb(i: int, total: int, name: str) -> None:
    pct = i / max(total, 1) * 100
    log.info(f"  🤖  [{pct:5.1f}%] {i:,}/{total:,} — {name[:55]}")


def _write_summary(stats: dict) -> None:
    lines = [
        "## ✦ ملخص نتائج محرك مهووس",
        "",
        "| المقياس | القيمة |",
        "|:---|---:|",
    ]
    for k, v in stats.items():
        lines.append(f"| {k} | {v} |")

    summary_text = "\n".join(lines)

    # GitHub Actions Step Summary
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(summary_text + "\n")

    # ملف محلي
    (OUTPUT_DIR / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
    log.info("📋 تم كتابة الملخص")


# ─── البرنامج الرئيسي ─────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    date_str = datetime.now().strftime("%Y-%m-%d")

    log.info("=" * 65)
    log.info("  ✦ محرك مهووس للحسم v10.0 — بدء التشغيل")
    log.info(f"  📅 التاريخ: {date_str}")
    log.info(f"  🤖 الذكاء الصناعي: {'مفعّل' if USE_LLM and ANTHROPIC_KEY else 'غير مفعّل'}")
    log.info(f"  🔄 وضع الأوصاف فقط: {'نعم' if DESCRIBE_ONLY else 'لا'}")
    log.info("=" * 65)

    # ── وضع استئناف الأوصاف فقط ──────────────────────────────────────────────
    if DESCRIBE_ONLY:
        _run_describe_only(date_str)
        return

    # ── 1. متجر مهووس ────────────────────────────────────────────────────────
    store_files = _load_csv_files(INPUT_STORE)
    if not store_files:
        log.error("❌ لا توجد ملفات متجر! ضع CSV في input/store/")
        sys.exit(1)

    log.info("📥 تحميل بيانات متجر مهووس...")
    store_df = load_store_products(store_files)
    if store_df.empty:
        log.error("❌ ملفات المتجر فارغة!")
        sys.exit(1)
    log.info(f"✅ {len(store_df):,} منتج في متجر مهووس")

    # ── 2. المنافسون ──────────────────────────────────────────────────────────
    comp_files = _load_csv_files(INPUT_COMP)
    if not comp_files:
        log.error("❌ لا توجد ملفات منافسين! ضع CSV في input/competitors/")
        sys.exit(1)

    log.info("📦 تحميل بيانات المنافسين...")
    comp_df = load_competitor_products(comp_files)
    if comp_df.empty:
        log.error("❌ ملفات المنافسين فارغة!")
        sys.exit(1)
    log.info(f"✅ {len(comp_df):,} منتج من {len(comp_files)} منافس")

    # تقليص حجم التشغيل إن طُلب
    if MAX_PRODUCTS and len(comp_df) > MAX_PRODUCTS:
        comp_df = comp_df.head(MAX_PRODUCTS)
        log.info(f"✂️  تم تقليص إلى {MAX_PRODUCTS:,} منتج (MAX_PRODUCTS)")

    # ── 3. الماركات (اختياري) ─────────────────────────────────────────────────
    brand_files = list(INPUT_BRANDS.glob("*.csv"))
    existing_brands = load_brands(brand_files[0]) if brand_files else []

    # ── 4. فهرس المتجر ────────────────────────────────────────────────────────
    log.info("🧠 بناء فهرس المتجر...")
    semantic_idx = SemanticIndex()
    semantic_idx.build(store_df, progress_cb=lambda m: log.info(f"  {m}"))
    log.info(f"✅ الفهرس جاهز: {len(semantic_idx.store_features):,} منتج")

    # ── 5. المحرك الرئيسي ─────────────────────────────────────────────────────
    engine = MahwousEngine(semantic_idx, existing_brands)
    log.info(f"⚖️ بدء التحليل الخماسي على {len(comp_df):,} منتج...")

    new_opps, duplicates, reviews, _ = engine.run(
        store_df=store_df,
        comp_df=comp_df,
        progress_cb=_progress_cb,
        log_cb=lambda m: log.info(m),
    )

    elapsed_match = time.time() - t0
    log.info("=" * 65)
    log.info(f"🎉 اكتملت المطابقة في {elapsed_match:.1f}ث")
    log.info(f"   🌟 فرص جديدة:     {len(new_opps):,}")
    log.info(f"   🚫 مكررات:        {len(duplicates):,}")
    log.info(f"   🔍 مراجعة يدوية:  {len(reviews):,}")
    log.info("=" * 65)

    # ── 6. توليد الأوصاف ──────────────────────────────────────────────────────
    if USE_LLM and ANTHROPIC_KEY and new_opps:
        log.info(f"🤖 بدء توليد الأوصاف لـ {len(new_opps):,} منتج جديد...")
        new_opps = generate_batch(
            new_opps,
            checkpoint_path=CHECKPOINT,
            progress_cb=_desc_progress_cb,
            requests_per_minute=40,
        )
        desc_count = sum(1 for r in new_opps if r.description)
        log.info(f"✅ تم توليد {desc_count:,} وصف")
    else:
        if not ANTHROPIC_KEY:
            log.info("⏭️ تخطي توليد الأوصاف (ANTHROPIC_API_KEY غير مضبوط)")
        elif not USE_LLM:
            log.info("⏭️ توليد الأوصاف مُعطّل (USE_LLM=false)")

    # ── 7. حفظ النتائج ────────────────────────────────────────────────────────
    log.info("💾 حفظ النتائج...")

    # ملف صلة الجاهز للرفع المباشر
    if new_opps:
        salla_bytes = export_salla_csv(new_opps)
        path = OUTPUT_DIR / f"منتجات_جديدة_صلة_{date_str}.csv"
        path.write_bytes(salla_bytes)
        log.info(f"  📄 {path.name} ({len(new_opps):,} منتج)")

    # ملف الفرص التفصيلي
    if new_opps:
        det_bytes = export_detailed_csv(new_opps)
        path = OUTPUT_DIR / f"فرص_جديدة_تفصيلي_{date_str}.csv"
        path.write_bytes(det_bytes)
        log.info(f"  📄 {path.name}")

    # ملف المكررات
    if duplicates:
        det_bytes = export_detailed_csv(duplicates)
        path = OUTPUT_DIR / f"مكررات_{date_str}.csv"
        path.write_bytes(det_bytes)
        log.info(f"  📄 {path.name} ({len(duplicates):,} منتج)")

    # ملف المراجعة اليدوية
    if reviews:
        det_bytes = export_detailed_csv(reviews)
        path = OUTPUT_DIR / f"مراجعة_يدوية_{date_str}.csv"
        path.write_bytes(det_bytes)
        log.info(f"  📄 {path.name} ({len(reviews):,} منتج)")

    # حذف checkpoint بعد النجاح
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()

    # ── 8. الملخص النهائي ─────────────────────────────────────────────────────
    total_time = time.time() - t0
    desc_count = sum(1 for r in new_opps if r.description)

    _write_summary({
        "📅 تاريخ التشغيل":        date_str,
        "🏪 منتجات متجرنا":        f"{len(store_df):,}",
        "🔍 منتجات المنافسين":      f"{len(comp_df):,}",
        "🌟 **فرص جديدة**":        f"**{len(new_opps):,}**",
        "📝 أوصاف مولّدة":         f"{desc_count:,}",
        "🚫 مكررات محظورة":        f"{len(duplicates):,}",
        "🔍 مراجعة يدوية":         f"{len(reviews):,}",
        "⏱️ وقت التشغيل":          f"{total_time:.0f} ثانية",
        "🤖 الذكاء الصناعي":       "نشط ✅" if ANTHROPIC_KEY else "غير مفعّل ⏭️",
    })

    log.info("=" * 65)
    log.info("✨ اكتمل بنجاح — قم بتحميل النتائج من Artifacts")
    log.info("=" * 65)


# ── وضع الأوصاف فقط (Describe Only) ─────────────────────────────────────────

def _run_describe_only(date_str: str) -> None:
    """يستأنف توليد الأوصاف من checkpoint موجود مسبقاً."""
    from logic import MatchResult

    if not CHECKPOINT.exists():
        log.error("❌ لا يوجد checkpoint! شغّل المحرك الكامل أولاً.")
        sys.exit(1)

    saved = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    all_results = [
        MatchResult(comp_name=e["comp_name"], description=e.get("description", ""))
        for e in saved
    ]
    pending = [r for r in all_results if not r.description]
    log.info(f"🔄 استئناف: {len(pending)} متبقٍ من {len(all_results)}")

    if pending:
        generate_batch(
            all_results,
            checkpoint_path=CHECKPOINT,
            progress_cb=_desc_progress_cb,
        )

    log.info("✅ اكتمل وضع الأوصاف فقط")


if __name__ == "__main__":
    main()
