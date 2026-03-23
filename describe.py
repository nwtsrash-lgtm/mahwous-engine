"""
describe.py — مولّد الأوصاف بالذكاء الصناعي (Claude API)
===========================================================
يولّد أوصاف منتجات عطور احترافية محسّنة لـ SEO باستخدام Claude Sonnet.
يعمل بشكل متزامن مع دعم retry واستئناف من النقطة التي توقف عندها.
"""

from __future__ import annotations
import os
import time
import json
import logging
from pathlib import Path
from typing import Optional

import anthropic

log = logging.getLogger("mahwous-describe")

# ─── إعداد العميل ────────────────────────────────────────────────────────────

def _get_client() -> Optional[anthropic.Anthropic]:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        log.warning("⚠️ ANTHROPIC_API_KEY غير مضبوط — سيتم تخطي توليد الأوصاف")
        return None
    return anthropic.Anthropic(api_key=key)


# ─── System Prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
أنت خبير عالمي في كتابة أوصاف منتجات العطور والجمال محسّنة لمحركات البحث (Google SEO) \
ومحركات بحث الذكاء الصناعي (GEO/AIO). تعمل حصرياً لمتجر "مهووس" (Mahwous) — \
الوجهة الأولى للعطور الفاخرة في السعودية.

## مهمتك
كتابة وصف منتج احترافي محسّن لـ SEO بـ 600-800 كلمة بالعربية الفصحى.

## البنية الإلزامية

# [اسم المنتج الكامل]

[فقرة افتتاحية عاطفية 80-100 كلمة. الكلمة الرئيسية في أول 50 كلمة. دعوة مبكرة للشراء من مهووس]

**تفاصيل المنتج:**
- **الماركة:** [الماركة]
- **المصمم:** [المصمم/الموقّع]
- **سنة الإصدار:** [السنة الحقيقية]
- **العائلة العطرية:** [خشبية/زهرية/شرقية/حمضية/مائية...]
- **التركيز:** [EDP/EDT/Parfum/Extrait...]
- **الحجم:** [X مل]
- **مناسب لـ:** [رجال/نساء/للجنسين]

## رحلة العطر: الهرم العطري الفاخر
**النفحات العليا (Top Notes):** [مكونات حقيقية من Fragrantica + وصف حسي]
**النفحات الوسطى (Heart Notes):** [مكونات + وصف]
**النفحات الأساسية (Base Notes):** [مكونات + ثبات X ساعات]

## لماذا تختار هذا المنتج من مهووس؟
- **[ميزة 1]:** ...
- **[ميزة 2]:** ...
- **[ميزة 3]:** ...
- **[ميزة 4]:** ...

## متى وأين ترتديه؟
[الفصول، الأوقات، المناسبات — 60-80 كلمة]

## لمسة خبير من مهووس
[تقييم احترافي بضمير "نحن"، ثبات بالساعات، فوحان، مقارنة، لمن نوصي به — 80-100 كلمة]

## الأسئلة الشائعة

**س: هل [اسم المنتج] مناسب للاستخدام اليومي؟**
ج: [إجابة 40-60 كلمة]

**س: ما مدة ثبات [اسم المنتج]؟**
ج: [أرقام دقيقة]

**س: هل يناسب الطقس الحار في السعودية؟**
ج: [إجابة مخصصة للسوق السعودي]

---
**عالمك العطري يبدأ من مهووس! اطلبه الآن بأفضل سعر مع ضمان الأصالة وتوصيل سريع.**

## قواعد صارمة
- مكونات عطرية حقيقية ومعروفة (من معرفتك بـ Fragrantica)
- لا إيموجي أبداً
- **Bold** للكلمات المفتاحية فقط
- أرقام الثبات والفوحان إلزامية
- أرسل الوصف مباشرةً بدون أي مقدمة أو تعليق\
"""


# ─── توليد وصف منتج واحد ────────────────────────────────────────────────────

def generate_description(
    client: anthropic.Anthropic,
    name: str,
    price: str,
    category: str,
    brand: str,
    image: str,
    retries: int = 3,
    delay: float = 5.0,
) -> str:
    prompt = (
        f"اسم المنتج: {name}\n"
        f"السعر: {price} ريال سعودي\n"
        f"التصنيف: {category}\n"
        f"الماركة: {brand or 'غير محدد'}\n"
        f"رابط الصورة: {image}"
    )

    for attempt in range(1, retries + 1):
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = next((b.text for b in msg.content if b.type == "text"), "")
            if text:
                return text
            raise ValueError("رد فارغ من الـ API")
        except anthropic.RateLimitError:
            wait = delay * (2 ** attempt)
            log.warning(f"  Rate limit — انتظار {wait:.0f}ث...")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            log.warning(f"  API خطأ ({e.status_code}) — محاولة {attempt}/{retries}")
            time.sleep(delay)
        except Exception as e:
            log.warning(f"  خطأ غير متوقع: {e} — محاولة {attempt}/{retries}")
            time.sleep(delay)

    return ""  # فشل بعد كل المحاولات


# ─── توليد دُفعة من المنتجات ─────────────────────────────────────────────────

def generate_batch(
    results: list,
    checkpoint_path: Path,
    progress_cb=None,
    requests_per_minute: int = 40,
) -> list:
    """
    يولّد الأوصاف لقائمة MatchResult ويحفظ checkpoint بعد كل منتج.
    يستأنف من آخر نقطة محفوظة إن وُجدت.
    """
    client = _get_client()
    if not client:
        log.info("⏭️ تخطي توليد الأوصاف (لا يوجد ANTHROPIC_API_KEY)")
        return results

    # استئناف من checkpoint
    done_names: set[str] = set()
    if checkpoint_path.exists():
        try:
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            for entry in saved:
                name = entry.get("comp_name", "")
                desc = entry.get("description", "")
                if desc:
                    # حقن الوصف في الكائنات الموجودة
                    for r in results:
                        if r.comp_name == name:
                            r.description = desc
                            break
                    done_names.add(name)
            log.info(f"📂 استئناف من checkpoint — {len(done_names)} منتج مكتمل مسبقاً")
        except Exception as e:
            log.warning(f"⚠️ تعذر قراءة checkpoint: {e}")

    pending = [r for r in results if r.comp_name not in done_names]
    total   = len(pending)
    log.info(f"🤖 توليد الأوصاف: {total} منتج متبقٍ (من {len(results)} إجمالاً)...")

    min_interval = 60.0 / requests_per_minute

    for i, result in enumerate(pending, 1):
        t_start = time.time()

        if progress_cb:
            progress_cb(i, total, result.comp_name)

        desc = generate_description(
            client,
            name=result.comp_name,
            price=result.comp_price,
            category=result.category,
            brand=result.brand,
            image=result.comp_image,
        )
        result.description = desc

        # حفظ checkpoint
        checkpoint_data = [
            {"comp_name": r.comp_name, "description": r.description}
            for r in results
            if r.description
        ]
        try:
            checkpoint_path.write_text(
                json.dumps(checkpoint_data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            log.warning(f"⚠️ فشل حفظ checkpoint: {e}")

        # تحكم في السرعة
        elapsed = time.time() - t_start
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    done_count = sum(1 for r in results if r.description)
    log.info(f"✅ اكتمل التوليد: {done_count}/{len(results)} وصف")
    return results
