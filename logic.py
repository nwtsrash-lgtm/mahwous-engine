"""
logic.py — محرك مهووس للحسم v10.0
====================================
المقارنة الخماسية الأبعاد: الاسم + الحجم + التركيز + النوع + FAISS
"""

from __future__ import annotations
import re
import io
import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rapidfuzz import process, fuzz

log = logging.getLogger("mahwous-logic")

# ─── 1. محرك الاستخراج والتفكيك ─────────────────────────────────────────────

def extract_attributes(name: str) -> dict:
    """تفكيك اسم المنتج إلى أبعاده الأساسية"""
    n = str(name).lower()

    # الحجم
    m = re.search(r'(\d+)\s*(مل|ml|جرام|g\b|oz|ملم)', n, re.IGNORECASE)
    size = int(m.group(1)) if m else 0

    # النوع
    product_type = "عطر تجاري"
    if any(w in n for w in ['تستر', 'tester', 'بدون كرتون', 'ديمو']):
        product_type = "تستر"
    elif any(w in n for w in ['طقم', 'مجموعة', 'set', 'gift']):
        product_type = "طقم هدايا"
    elif any(w in n for w in ['عطر شعر', 'للشعر', 'hair mist']):
        product_type = "عطر شعر"
    elif any(w in n for w in ['لوشن', 'كريم جسم', 'lotion', 'body cream']):
        product_type = "عناية (لوشن/كريم)"
    elif any(w in n for w in ['جل استحمام', 'شاور جل', 'shower gel']):
        product_type = "شاور جل"
    elif any(w in n for w in ['معطر جسم', 'بدي مست', 'body mist']):
        product_type = "معطر جسم"
    elif any(w in n for w in ['مزيل عرق', 'ديودرنت', 'deodorant', 'stick']):
        product_type = "مزيل عرق"

    # التركيز
    conc = "غير محدد"
    if any(w in n for w in ['اكستريت', 'extrait']):
        conc = "Extrait"
    elif any(w in n for w in ['او دي بارفيوم', 'او دو بارفيوم', 'edp', 'eau de parfum', 'بارفيوم', 'برفيوم']):
        conc = "EDP"
    elif any(w in n for w in ['او دي تواليت', 'او دو تواليت', 'edt', 'eau de toilette', 'تواليت']):
        conc = "EDT"
    elif any(w in n for w in ['بارفان', 'parfum', 'pure parfum']):
        conc = "Parfum"
    elif any(w in n for w in ['كولونيا', 'cologne', 'edc']):
        conc = "EDC"

    if 'انتنس' in n or 'intense' in n: conc += " Intense"
    if 'ابسولو' in n or 'absolu' in n: conc += " Absolu"

    # الاسم النقي
    clean = re.sub(r'\d+\s*(مل|ml|جرام|g\b|oz|ملم|x)', '', n)
    for w in ['عطر', 'او دي بارفيوم', 'او دو بارفيوم', 'او دي تواليت', 'او دو تواليت',
              'بارفيوم', 'برفيوم', 'تواليت', 'اكستريت', 'بارفان', 'كولونيا',
              'تستر', 'طقم', 'مجموعة', 'للرجال', 'للنساء', 'نسائي', 'رجالي',
              'edp', 'edt', 'tester', 'set', 'hair mist']:
        clean = clean.replace(w, '')
    clean = ' '.join(clean.split())

    return {'size': size, 'type': product_type, 'concentration': conc, 'clean_name': clean}


# ─── 2. تجهيز البيانات ──────────────────────────────────────────────────────

class FeatureParser:
    @staticmethod
    def extract_features(df: pd.DataFrame, source_type: str = "store") -> pd.DataFrame:
        out = pd.DataFrame()
        if source_type == "store":
            name_col  = next((c for c in df.columns if 'اسم' in str(c) or 'أسم' in str(c)), df.columns[2] if len(df.columns) > 2 else None)
            price_col = next((c for c in df.columns if 'سعر' in str(c)), df.columns[7] if len(df.columns) > 7 else None)
            img_col   = next((c for c in df.columns if 'صورة' in str(c)), df.columns[4] if len(df.columns) > 4 else None)
        else:
            name_col  = next((c for c in df.columns if 'name' in str(c).lower() or 'اسم' in str(c) or 'pakbB' in str(c)), df.columns[2] if len(df.columns) > 2 else None)
            price_col = next((c for c in df.columns if 'price' in str(c).lower() or 'سعر' in str(c) or 'sm' in str(c).lower()), df.columns[3] if len(df.columns) > 3 else None)
            img_col   = next((c for c in df.columns if 'src' in str(c).lower() or 'صورة' in str(c)), df.columns[1] if len(df.columns) > 1 else None)

        out['orig_name'] = df[name_col].astype(str) if name_col else ""
        out['price']     = df[price_col].astype(str) if price_col else "0"
        out['image']     = df[img_col].astype(str)   if img_col   else ""
        if 'source_file' in df.columns:
            out['source_file'] = df['source_file']

        parsed = out['orig_name'].apply(extract_attributes)
        out['size']          = [p['size']          for p in parsed]
        out['type']          = [p['type']           for p in parsed]
        out['concentration'] = [p['concentration']  for p in parsed]
        out['clean_name']    = [p['clean_name']     for p in parsed]
        return out


class SemanticIndex:
    def __init__(self, model=None):
        self.model = model
        self.store_features = pd.DataFrame()

    def build(self, df: pd.DataFrame, progress_cb=None):
        if progress_cb: progress_cb("جاري تفكيك منتجات المتجر...")
        self.store_features = FeatureParser.extract_features(df, "store")


# ─── 3. نموذج النتيجة ───────────────────────────────────────────────────────

@dataclass
class MatchResult:
    comp_name:    str = ""
    comp_image:   str = ""
    comp_price:   str = ""
    comp_source:  str = ""
    store_name:   str = ""
    confidence:   float = 0.0
    layer_used:   str = ""
    brand:        str = ""
    verdict:      str = ""
    reason:       str = ""
    description:  str = ""   # الوصف المولّد بالذكاء الاصطناعي
    category:     str = ""


# ─── 4. المحرك الرئيسي ──────────────────────────────────────────────────────

class MahwousEngine:
    def __init__(self, semantic_index: SemanticIndex, brands_list: list, gemini_oracle=None):
        self.semantic_index = semantic_index
        self.brands_list    = brands_list
        self.gemini_oracle  = gemini_oracle

    def run(self, store_df, comp_df, use_llm=False, progress_cb=None, log_cb=None):
        if log_cb: log_cb("بدء المطابقة خماسية الأبعاد...")

        new_opps, duplicates, reviews = [], [], []
        comp_features  = FeatureParser.extract_features(comp_df, "comp")
        store_feats    = self.semantic_index.store_features
        store_clean    = {i: row['clean_name'] for i, row in store_feats.iterrows()}
        total = len(comp_features)

        for i, row in comp_features.iterrows():
            comp_orig = str(row['orig_name']).strip()
            if comp_orig in ('nan', '') or not comp_orig: continue
            if progress_cb: progress_cb(i + 1, total, comp_orig[:50])

            best = process.extractOne(row['clean_name'], store_clean, scorer=fuzz.token_set_ratio)
            score, verdict, reason = 0.0, "فرصة جديدة", "منتج جديد تماماً"

            if best:
                _, match_score, match_idx = best
                score = match_score / 100.0
                store_row = store_feats.iloc[match_idx]

                if score >= 0.88:
                    if row['type'] != store_row['type']:
                        reason = f"اختلاف النوع: لدينا ({store_row['type']}) والمنافس ({row['type']})"
                    elif row['size'] != store_row['size'] and row['size'] != 0 and store_row['size'] != 0:
                        reason = f"اختلاف الحجم: لدينا ({store_row['size']}مل) والمنافس ({row['size']}مل)"
                    elif (row['concentration'] != store_row['concentration']
                          and row['concentration'] != 'غير محدد'
                          and store_row['concentration'] != 'غير محدد'):
                        reason = f"اختلاف التركيز: لدينا ({store_row['concentration']}) والمنافس ({row['concentration']})"
                    else:
                        verdict = "مكرر"
                        reason  = "تطابق تام (الاسم + الحجم + التركيز + النوع)"
                elif score >= 0.60:
                    verdict = "مراجعة يدوية"
                    reason  = f"تشابه في الاسم ({match_score:.0f}%) — يرجى التأكد"

            result = MatchResult(
                comp_name=comp_orig, comp_price=str(row['price']),
                comp_image=str(row['image']),
                comp_source=str(row.get('source_file', '')),
                store_name=store_feats.iloc[best[2]]['orig_name'] if best and score >= 0.60 else '',
                confidence=round(score * 100, 1),
                layer_used="5D-Analyzer", verdict=verdict, reason=reason,
                category=_guess_category(comp_orig),
                brand=_guess_brand(comp_orig),
            )

            if verdict == "مكرر":         duplicates.append(result)
            elif verdict == "فرصة جديدة": new_opps.append(result)
            else:                          reviews.append(result)

        return new_opps, duplicates, reviews, []


# ─── 5. التصميم المساعد ─────────────────────────────────────────────────────

def _guess_category(name: str) -> str:
    n = name.lower()
    if any(w in n for w in ['طقم', 'مجموعة', 'set']): return 'طقم وهدايا'
    if any(w in n for w in ['مزيل', 'ديودرنت']):       return 'مزيل عرق'
    if any(w in n for w in ['لوشن', 'كريم']):           return 'عناية بالجسم'
    if any(w in n for w in ['شاور', 'جل']):             return 'عناية بالجسم'
    if any(w in n for w in ['معطر جسم', 'بدي']):        return 'معطرات جسم'
    if any(w in n for w in ['عطر شعر', 'hair']):        return 'عطور شعر'
    return 'عطور'

KNOWN_BRANDS = [
    'توم فورد','باكو رابان','ديور','شانيل','كارولينا هيريرا','جيفنشي',
    'جورجيو ارماني','هوغو بوس','مونت بلانك','لانكوم','بولغاري','فالنتينو',
    'فيرساتشي','روبرتو كافالي','كلفن كلاين','بربري','هيرمس','كلوي',
    'ايف سان لوران','اكوا دي بارما','نارسيسو','ميزون مارجيلا',
    'لا كوست','دولتشي غابانا','ديزل','كيلي','لانفن',
]

def _guess_brand(name: str) -> str:
    for b in KNOWN_BRANDS:
        if b in name.lower(): return b
    return ''


# ─── 6. تحميل الملفات ───────────────────────────────────────────────────────

def load_store_products(files: list) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            path = str(f)
            if path.endswith('.csv'):
                df = pd.read_csv(f, encoding='utf-8-sig', skiprows=1)
            else:
                df = pd.read_excel(f)
            frames.append(df)
        except Exception as e:
            log.warning(f"تعذر قراءة {f}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_competitor_products(files: list) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            path = str(f)
            if path.endswith('.csv'):
                df = pd.read_csv(f, encoding='utf-8-sig')
            else:
                df = pd.read_excel(f)
            df['source_file'] = Path(f).stem
            frames.append(df)
        except Exception as e:
            log.warning(f"تعذر قراءة {f}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_brands(file) -> list:
    return []


# ─── 7. التصدير بتنسيق صلة (سلة) ────────────────────────────────────────────

SALLA_HEADER_1 = ['بيانات المنتج'] + ['']*39
SALLA_HEADER_2 = [
    'النوع','أسم المنتج','تصنيف المنتج','صورة المنتج','وصف صورة المنتج',
    'نوع المنتج','سعر المنتج','الوصف','هل يتطلب شحن؟','رمز المنتج sku',
    'سعر التكلفة','السعر المخفض','تاريخ بداية التخفيض','تاريخ نهاية التخفيض',
    'اقصي كمية لكل عميل','إخفاء خيار تحديد الكمية','اضافة صورة عند الطلب',
    'الوزن','وحدة الوزن','الماركة','العنوان الترويجي','تثبيت المنتج','الباركود',
    'السعرات الحرارية','MPN','GTIN','خاضع للضريبة ؟','سبب عدم الخضوع للضريبة',
    '[1] الاسم','[1] النوع','[1] القيمة','[1] الصورة / اللون',
    '[2] الاسم','[2] النوع','[2] القيمة','[2] الصورة / اللون',
    '[3] الاسم','[3] النوع','[3] القيمة','[3] الصورة / اللون',
]


def export_salla_csv(results: list[MatchResult]) -> bytes:
    if not results: return b""
    rows = [SALLA_HEADER_1, SALLA_HEADER_2]
    for i, r in enumerate(results, 1):
        sku = f"MHW-{i:04d}"
        row = [
            'منتج', r.comp_name, r.category, r.comp_image, r.comp_name,
            'منتج جاهز', r.comp_price, r.description or r.reason,
            'نعم', sku, '', '', '', '', '0', '', '0.2', 'كجم',
            r.brand, '', '', '', '', '', '', 'نعم',
            '', '', '', '', '', '', '', '', '', '', '', '',
        ]
        rows.append(row)
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, encoding='utf-8-sig')
    return ('\ufeff' + buf.getvalue()).encode('utf-8')


def export_detailed_csv(results: list[MatchResult]) -> bytes:
    if not results: return b""
    data = [{
        'اسم منتج المنافس':  r.comp_name,
        'صورة المنافس':      r.comp_image,
        'سعر المنافس':       r.comp_price,
        'مصدر الملف':        r.comp_source,
        'التصنيف':           r.category,
        'الماركة':           r.brand,
        'أقرب منتج لدينا':   r.store_name,
        'نسبة التطابق %':    r.confidence,
        'القرار':            r.verdict,
        'سبب القرار':        r.reason,
        'الوصف المولّد':     r.description,
    } for r in results]
    df = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding='utf-8-sig')
    return ('\ufeff' + buf.getvalue()).encode('utf-8')


def export_brands_csv(brands: list) -> bytes:
    return b""
