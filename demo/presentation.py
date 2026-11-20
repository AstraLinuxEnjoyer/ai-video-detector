"""Human-facing score units; never used to make an inference decision."""
import math


def signal_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        return '—'
    return f'{value * 100:.1f}'.rstrip('0').rstrip('.').replace('.', ',')


def signal(value):
    text = signal_number(value)
    return 'Нет оценки' if text == '—' else f'{text} из 100'


def points(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return 'нет данных'
    return f'{value * 100:.1f}'.rstrip('0').rstrip('.').replace('.', ',') + ' п. шкалы'


def signal_level(value, threshold=.65, review=.35):
    if signal_number(value) == '—':
        return 'Нет оценки'
    if value >= threshold:
        return 'Сильный сигнал'
    return 'Повышенный сигнал' if value >= min(review, threshold) else 'Слабый сигнал'
