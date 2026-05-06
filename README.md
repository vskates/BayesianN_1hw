# BayesianN_1hw

Проект по задаче «Байес-оптимальный морской бой».

## Что внутри

- `battleship.py` — генерация досок, игровой движок, стратегии и SVG-визуализация.
- `run_analysis.py` — запуск экспериментов и генерация итоговых файлов в `outputs/`.
- `test_battleship.py` — базовые тесты корректности.
- `REPORT.md` — теоретическое объяснение и выводы.

## Как запустить

```bash
python3 -m unittest -v
python3 run_analysis.py
```

После запуска появятся:

- `outputs/example_heatmap.svg`
- `outputs/shot_distribution.svg`
- `outputs/summary.json`
