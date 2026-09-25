# LinkReview

- Здесь собраны основные работы, на которые опирается обзор литературы.
- Источники разделены по смысловым блокам: классические модели поверхности, глубокое обучение, динамическое прогнозирование.
- Таблица будет расширяться по мере развития работы.

> [!NOTE]
> Это рабочая версия обзора литературы. Перед финальной версией ВКР список будет дополнен статьями по данным Московской биржи, микроструктуре российского рынка опционов и методам проверки арбитражных ограничений.

| Topic | Title | Year | Authors | Paper | Code | Summary |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| Классические модели | Managing Smile Risk | 2002 | Hagan et al. | [Paper](https://www.next-finance.net/IMG/pdf/pdf_SABR.pdf) | — | Классическая модель SABR для описания улыбки подразумеваемой волатильности. Используется как один из базовых параметрических подходов. |
| Классические модели | Arbitrage-Free SVI Volatility Surfaces | 2014 | Gatheral, Jacquier | [DOI](https://doi.org/10.1080/14697688.2013.819986) | — | Формулирует условия построения SVI-поверхности без статического арбитража. Важный теоретический базовый метод для сравнения. |
| Глубокое обучение | Deep Smoothing of the Implied Volatility Surface | 2020 | Ackerer, Tagasovska, Vatter | [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/858e47701162578e5e627cd93ab0938a-Abstract.html) | — | Нейросетевое сглаживание поверхности подразумеваемой волатильности, работа с разреженными и ошибочными котировками, штрафы за статический арбитраж. Одна из наиболее близких к нашей постановке работ. |
| Глубокое обучение | Deep Learning Volatility: A Deep Neural Network Perspective on Pricing and Calibration in (Rough) Volatility Models | 2021 | Horvath, Muguruza, Tomas | [DOI](https://doi.org/10.1080/14697688.2020.1817974) | — | Показывает применение глубоких нейронных сетей для ускорения калибровки моделей волатильности. Нужна для обоснования применения нейросетевых методов в задачах производных инструментов. |
| Восстановление поверхности | Volatility Surface Reconstruction using Deep Learning under No-Arbitrage Constraints | 2026 | Rodriguez Manzi | [arXiv](https://arxiv.org/abs/2605.24031) | — | Сравнивает несколько архитектур глубокого обучения для восстановления поверхности по разреженным и шумным данным и исследует влияние безарбитражных ограничений. |
| Прогнозирование поверхности | Hexagon-Net: Heterogeneous Cross-View Aligned Graph Attention Networks for Implied Volatility Surface Prediction | 2025 | Liang et al. | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5277958) | — | Графовая модель прогнозирования поверхности, учитывающая связи между различными опционными контрактами. |
| Смена рыночных режимов | Continual Learning for Implied Volatility Surfaces under Regime Shifts | 2026 | Zhuang et al. | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1544612326005751) | — | Исследует адаптацию модели поверхности к смене режимов волатильности и проблему сохранения знаний о предыдущих режимах. |
| Генеративные модели | Arbitrage-Aware Multi-Step Forecasting of Implied Volatility Surfaces: Modelling Surface Trajectories Using Latent Diffusion | 2026 | Buchegger, Gonon | [arXiv](https://arxiv.org/abs/2608.22478) | — | Современная работа по мног шаговому прогнозированию траекторий поверхности с использованием диффузионной модели и контролем арбитражных нарушений. |
