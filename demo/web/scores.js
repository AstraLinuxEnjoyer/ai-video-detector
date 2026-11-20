/* Presentation only: inference, thresholds and stored scores stay in 0–1. */
(() => {
  const number = new Intl.NumberFormat('ru-RU', {maximumFractionDigits:1});
  const valid = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
  const units = value => valid(value) ? number.format(value * 100) : '—';
  const format = value => valid(value) ? `${units(value)} из 100` : 'Нет оценки';
  const points = value => {
    if (!Number.isFinite(value)) return 'нет данных';
    const value100 = value * 100;
    const forms = {one:'пункт',few:'пункта',many:'пунктов',other:'пункта'};
    return `${number.format(value100)} ${forms[new Intl.PluralRules('ru-RU').select(value100)]}`;
  };
  function level(value, threshold = .65, review = .35) {
    if (!valid(value)) return {key:'missing',label:'Нет оценки'};
    if (valid(threshold) && value >= threshold) return {key:'strong',label:'Сильный сигнал'};
    if (value >= Math.min(review, threshold ?? review)) return {key:'raised',label:'Повышенный сигнал'};
    return {key:'weak',label:'Слабый сигнал'};
  }
  const api = {valid,units,format,points,level};
  if (typeof window !== 'undefined') window.TraceScores = api;
  if (typeof module !== 'undefined') module.exports = api;
})();
