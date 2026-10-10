"""Контекстная расстановка ударений для русского текста.

Два уровня:
  1. ``ruaccent`` (если установлен) — нейросетевой постановщик ударений,
     выбирающий вариант по контексту предложения («до́мой» vs «домá»).
     Ударение проставляется combining-символом U+0301 после гласной.
  2. Словарь омографов (:data:`HOMOGRAPHS`) — ручной fallback для слов,
     где даже ruaccent ошибается, плюс быстрые точные правила.

XTTS (espeak-ng) понимает combining-ударение U+0301, но плохо относится
к букве «Ё», поэтому перед инференсом текст проходит :func:`prepare_for_tts`:
  • «Ё/ё» → «Е/е» (ударная «ё» уже помечена U+0301 — чтение не теряется);
  • NFC-нормализация и схлопывание двойных combining-символов.
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger("reader.stress")

STRESS = "\u0301"          # combining acute accent
_VOWELS = "аеёиоуыэюя"

try:
    from ruaccent import RUAccent  # type: ignore

    _ruaccent = RUAccent(overlap_segment=False, segment_work=True,
                         word_optimizer=True, base_linguist=True)
    _ruaccent.load()
    HAS_RUACCENT = True
except Exception as _e:  # ImportError или ошибка загрузки моделей
    _ruaccent = None
    HAS_RUACCENT = False
    logger.debug("ruaccent недоступен: %s", _e)

# ------------------------------------------------------------------ Омографы
# Слово (без ударения, нижний регистр) → {вариант с U+0301: набор слов-подсказок
# контекста}. Если ни одна подсказка не найдена вокруг слова — берётся ключ "*".
# Подсказки проверяются в соседних ~6 словах.
HOMOGRAPHS: dict[str, dict[str, tuple[str, ...]]] = {
    "мука": {"му\u0301ка": ("тесто", "хлеб", "замес", "муки", "пшеничн"),
             "мук\u0301а": ("страд", "пытк", "адск", "му\u0301чит", "терпен")},
    "кружка": {"кру\u0301жка": ("вод", "пив", "ча", "глин", "желез", "метал"),
               "кружа\u0301": ("бег", "бежа\u0301л", "бега\u0301л", "круг",
                               "окрест", "рядом", "вокруг")},
    "кружки": {"кру\u0301жки": ("спортив", "секци", "класс", "детск"),
               "кружа\u0301": ("бег", "бежа\u0301л", "бега\u0301л", "круг",
                               "окрест", "рядом", "вокруг")},
    "кружку": {"кру\u0301жку": ("вод", "пив", "ча", "глин", "желез", "метал"),
               "кружа\u0301": ("бег", "бежа\u0301л", "бега\u0301л", "круг",
                               "окрест", "рядом", "вокруг")},
    "кружкой": {"кру\u0301жкой": ("вод", "пив", "ча", "поить"),
                "кружа\u0301": ("бег", "бежа\u0301л", "бега\u0301л", "круг",
                                "окрест", "рядом", "вокруг")},
    "сохнет": {"со\u0301хнет": ("белье", "бельё", "поленья", "дрова", "трава",
                                "сено", "волос", "крашен"),
                "сохне\u0301т": ("вид", "взгляд", "слух", "память", "кожей",
                                 "от", "из-за")},
    "мокнет": {"мо\u0301кнет": ("дожд", "луж", "вода", "плащ", "ткан"),
               "мокне\u0301т": ("вид", "взгляд", "слух", "от", "из-за")},
    "устойчивый": {"усто\u0301йчивый": ("положен", "стул", "конструкц", "позиц",
                                          "покров", "тон"),
                   "устойчи\u0301вый": ("человек", "характер", "критик",
                                         "результат", "к работе")},
    "свойственно": {"сво\u0301йственно": ("это", "ему", "ей", "для", "человеку"),
                    "свойстве\u0301нно": ("сказано", "произнесено", "тихо",
                                          "спокойно", "прошептала")},
    "ведомый": {"ве\u0301домый": ("опыт", "извест"),
                "ведо\u0301мый": ("руководств", "инстинкт", "огонь", "свет")},
    # «дома» — классический омограф: до́ма (дома, нахожусь) vs домá (направл.).
    "дома": {
        "до\u0301ма": ("раньше", "обычно", "часто", "бывало", "никогда", "почти",
                        "всегда", "утром", "вечером", "сидел", "остался", "был",
                        "была", "шли", "вернулись", "пока", "лежал", "спал",
                        "ждал", "работал", "читал", "смотрел", "живу", "живешь",
                        "иногда", "редко"),
        "дом́а": ("войти", "вошёл", "вошла", "зашёл", "зашла", "пришёл",
                        "пришла", "вернуться", "вернулся", "вернулась", "войди",
                        "заглянуть", "заглянул", "пробрался", "попасть", "попал",
                        "прятаться", "спрятаться", "заперты", "чужие", "пустые",
                        "знакомые", "родные", "его", "её", "наш", "наше", "ваш",
                        "мой", "моя", "твои", "чужой", "как", "будто", "словно"),
    },
    # «домой»: до́мой (нареч.-сравн. «как дома») vs домóй (направление).
    "домой": {
        "до\u0301мой": ("дым", "тень", "запах", "голос", "звук", "свет",
                         "ощущение", "помнится", "тянет", "манит", "хочется",
                         "мысли", "возвращение", "путь", "дорога", "обрёл",
                         "обрела", "кажется", "будто"),
        "дом\u0301ой": ("вернуться", "пойти", "пошёл", "пошла", "отправиться",
                         "собраться", "нужно", "пора", "хочу", "время", "бежать",
                         "поскорее"),
    },
}

# Быстрые контекстные замены «Ё→Е» со снятием неоднозначности.
_YO_FIXES: tuple[tuple[str, str], ...] = (
    ("все ещё", "все\u0301 ещё"),
    ("Всё ещё", "Все\u0301 ещё"),
    ("всё же", "все\u0301 же"),
    ("Всё же", "Все\u0301 же"),
)


def strip_stress(text: str) -> str:
    """Убирает combining-ударения (U+0301) из текста."""
    return text.replace(STRESS, "")


def has_stress(text: str) -> bool:
    return STRESS in text


def _word_variants(word: str) -> list[str]:
    """Возвращает возможные stressed-формы слова по словарю омографов."""
    base = strip_stress(word).lower().replace("ё", "е")
    entry = HOMOGRAPHS.get(base)
    if not entry:
        return []
    return list(entry.keys())


def apply_homographs(sentence: str) -> str:
    """Разрешает омографы в предложении по контекстным подсказкам."""
    tokens = re.split(r"(\s+)", sentence)
    words_idx = [i for i, t in enumerate(tokens)
                 if t and not t.isspace() and strip_stress(t).lower().replace("ё", "е")
                 in HOMOGRAPHS]
    if not words_idx:
        return sentence

    out = list(tokens)
    for wi in words_idx:
        raw = out[wi]
        core = re.sub(r"[^\w]", "", raw, flags=re.UNICODE)
        base = strip_stress(core).lower().replace("ё", "е")
        entry = HOMOGRAPHS[base]

        # окно контекста: ±6 слов
        neighbors: list[str] = []
        for j in range(max(0, wi - 12), min(len(tokens), wi + 13)):
            if j == wi or not tokens[j] or tokens[j].isspace():
                continue
            neighbors.append(strip_stress(tokens[j]).lower())
        ctx = " ".join(neighbors)

        # выбираем вариант по числу попавших в контекст подсказок
        best: str | None = None
        best_hits = 0
        for variant, hints in entry.items():
            hits = sum(1 for h in hints if h in ctx)
            if hits > best_hits:
                best, best_hits = variant, hits
        if best is not None:
            chosen = best
        elif "*" in entry:
            chosen = entry["*"]
        else:
            continue

        stressed = chosen
        if core[:1].isupper():
            stressed = stressed.capitalize()
        out[wi] = raw.replace(core, stressed)
        logger.debug("Омограф %r → %r (контекст-окно %d слов)",
                     core, stressed, 6)
    return "".join(out)


def stress_text(text: str) -> str:
    """Расставляет ударения по контексту.

    Порядок: сначала точные словарь-правила (омографы), затем ruaccent
    добирает остальные слова. Возвращает текст с U+0301.
    """
    if not text.strip():
        return text

    result_lines: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            result_lines.append(line)
            continue
        stressed = apply_homographs(line)
        if HAS_RUACCENT and _ruaccent is not None:
            try:
                stressed = _ruaccent.do(stressed)
            except Exception:
                logger.exception("ruaccent упал на строке, использую только словарь")
        result_lines.append(stressed)

    out = "\n".join(result_lines)
    for src, dst in _YO_FIXES:
        out = out.replace(src, dst)
    n = out.count(STRESS)
    logger.info("Ударения проставлены: %d слогов отмечено (ruaccent=%s)",
                n, "да" if HAS_RUACCENT else "нет")
    return out


def prepare_for_tts(text: str) -> str:
    """Финальная нормализация непосредственно перед инференсом XTTS.

    • Ё → Е (espeak-ng не читает «Ё», а U+0301 он понимает);
    • NFC, чтобы U+0301 оставался привязан к гласной;
    • защита от двойных combining-символов.
    """
    text = text.replace("Ё", "Е").replace("ё", "е")
    text = unicodedata.normalize("NFC", text)
    # схлопываем идущие подряд combining-акценты в один
    text = re.sub(f"({STRESS})+", STRESS, text)
    return text
