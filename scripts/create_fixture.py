"""Write original synthetic examples, with independently authored split phrasings.

This is a pipeline fixture, not enough data for a useful support model. No real
customer records are used. Expand with consented, reviewed data before training
for an actual deployment. Format: language | urgency | text.
"""

import json
from pathlib import Path

from support_lora.data import RESPONSES, validate_splits

EXAMPLES = {
    "account": {
        "train": [
            "en|normal|I forgot my account password and need to reset it.",
            "en|normal|How can I update the email linked to my profile?",
            "en|high|Someone accessed my account without permission today.",
            "en|high|An attacker changed my login and locked me out.",
            "ar|normal|نسيت كلمة المرور وأحتاج إلى استعادة حسابي.",
            "ar|normal|كيف أغير البريد الإلكتروني المرتبط بالحساب؟",
            "ar|high|شخص غريب دخل حسابي دون إذني اليوم.",
            "ar|high|تم اختراق حسابي وتغيير بيانات الدخول.",
        ],
        "validation": [
            "en|normal|The password reset email has not arrived.",
            "en|high|I received a login alert for a device I do not recognize.",
            "ar|normal|لم تصلني رسالة إعادة تعيين كلمة السر.",
            "ar|high|وصلني تنبيه بدخول غير معروف إلى الحساب.",
        ],
        "test": [
            "en|normal|My old phone number prevents me from signing in.",
            "en|high|My profile now shows a stranger's recovery email.",
            "ar|normal|رقم هاتفي القديم يمنعني من تسجيل الدخول.",
            "ar|high|وجدت بريداً غريباً في إعدادات استعادة حسابي.",
            "mixed|normal|محتاج أعمل reset password للحساب.",
            "mixed|high|My account اتسرق وحد غير الإيميل.",
        ],
    },
    "billing": {
        "train": [
            "en|normal|Please send an invoice for my purchase.",
            "en|normal|My payment was declined at checkout.",
            "en|high|I was charged twice for the same order.",
            "en|high|There is an unauthorized charge on my card.",
            "ar|normal|أحتاج فاتورة للطلب الذي دفعت قيمته.",
            "ar|normal|تم رفض عملية الدفع عند إتمام الشراء.",
            "ar|high|تم خصم المبلغ مرتين لنفس الطلب.",
            "ar|high|هناك عملية سحب غير مصرح بها من بطاقتي.",
        ],
        "validation": [
            "en|normal|Where can I download my payment receipt?",
            "en|high|Two identical transactions appeared on my bank statement.",
            "ar|normal|أين أجد إيصال الدفع الإلكتروني؟",
            "ar|high|ظهرت عمليتا دفع متطابقتان في كشف البنك.",
        ],
        "test": [
            "en|normal|Can you add my company tax number to the invoice?",
            "en|high|My card was billed for an order I never placed.",
            "ar|normal|هل يمكن إضافة الرقم الضريبي للشركة إلى الفاتورة؟",
            "ar|high|تم تحميل بطاقتي قيمة طلب لم أقم به.",
            "mixed|normal|عايز invoice فيها اسم الشركة.",
            "mixed|high|اتخصم مني payment مرتين بالغلط.",
        ],
    },
    "refund": {
        "train": [
            "en|normal|I would like to return an unused item for a refund.",
            "en|normal|What is the refund policy for a recent purchase?",
            "en|high|The return deadline is today and I need urgent help.",
            "en|high|My approved refund is overdue and I need urgent assistance.",
            "ar|normal|أريد إرجاع المنتج غير المستخدم واسترداد المبلغ.",
            "ar|normal|ما سياسة استرجاع مشتريات الأسبوع الماضي؟",
            "ar|high|تنتهي مهلة إرجاع المنتج اليوم وأحتاج مساعدة عاجلة.",
            "ar|high|تأخر استرداد المبلغ الموافق عليه وأحتاج متابعة عاجلة.",
        ],
        "validation": [
            "en|normal|Can I send this unopened product back?",
            "en|high|My last eligible day to request a return ends in hours.",
            "ar|normal|هل يمكن إعادة المنتج وهو ما زال مغلقاً؟",
            "ar|high|تبقت ساعات على آخر يوم مسموح لإرجاع مشترياتي.",
        ],
        "test": [
            "en|normal|The shoes do not fit; how do I get my money back?",
            "en|high|Support promised to investigate my delayed reimbursement urgently.",
            "ar|normal|مقاس الحذاء غير مناسب، كيف أستعيد نقودي؟",
            "ar|high|أحتاج تصعيداً عاجلاً بسبب تأخر تعويض مبلغ الإرجاع.",
            "mixed|normal|عايز أرجع المنتج وأعمل refund.",
            "mixed|high|آخر يوم للـ return النهارده، ساعدني بسرعة.",
        ],
    },
    "shipping": {
        "train": [
            "en|normal|Where is my parcel and what is the tracking number?",
            "en|normal|Can I change my delivery address before shipping?",
            "en|high|My package is marked delivered but has been stolen.",
            "en|high|The courier lost my time-sensitive delivery and I need urgent help.",
            "ar|normal|أين شحنتي وما رقم التتبع الخاص بها؟",
            "ar|normal|هل أستطيع تغيير عنوان التوصيل قبل الشحن؟",
            "ar|high|الشحنة مسجلة كمستلمة لكنها سُرقت من باب المنزل.",
            "ar|high|فقدت شركة الشحن طردي العاجل وأحتاج تدخلاً فورياً.",
        ],
        "validation": [
            "en|normal|When will the courier bring my order?",
            "en|high|A delivery photo shows my parcel at an unknown house; investigate urgently.",
            "ar|normal|متى يصل مندوب التوصيل بالطلب؟",
            "ar|high|صورة التسليم لبيت آخر وأحتاج تحقيقاً عاجلاً في ضياع الشحنة.",
        ],
        "test": [
            "en|normal|The tracking page has not updated since dispatch.",
            "en|high|The carrier confirmed my parcel is missing and it must be escalated now.",
            "ar|normal|لم تتحدث صفحة تتبع الطرد منذ خروجه من المخزن.",
            "ar|high|أكد الناقل فقدان الطرد وأطالب بالتصعيد الفوري.",
            "mixed|normal|ممكن tracking number للشحنة؟",
            "mixed|high|الـ courier ضيع الطرد ومحتاج حل فوراً.",
        ],
    },
}


def main():
    directory = Path(__file__).resolve().parents[1] / "data"
    directory.mkdir(exist_ok=True)
    for split in ("train", "validation", "test"):
        rows = []
        for category, splits in EXAMPLES.items():
            for index, value in enumerate(splits[split]):
                language, urgency, text = value.split("|", 2)
                rows.append(
                    {
                        "id": f"{split}-{category}-{index}",
                        "language": language,
                        "text": text,
                        "category": category,
                        "urgency": urgency,
                        "response": RESPONSES["en" if language == "en" else "ar"][category],
                    }
                )
        (directory / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
    print(validate_splits(directory))


if __name__ == "__main__":
    main()
