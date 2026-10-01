# Temporal and Cross-Contract Modeling of Sparse Implied Volatility Surfaces

[![License](https://badgen.net/github/license/masbalt/diplom?color=green)](https://github.com/masbalt/diplom/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/masbalt/diplom)](https://github.com/masbalt/diplom/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/masbalt/diplom.svg?color=0088ff)](https://github.com/masbalt/diplom/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/masbalt/diplom.svg?color=7f29d6)](https://github.com/masbalt/diplom/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Name Surname </td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Name Surname, PhD/DSc </td>
    </tr>
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td> Name Surname, PhD/DSc </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/main.pdf)

## Abstract

Implied volatility surfaces are widely used for option valuation, hedging, and risk management, but on less liquid markets the observable option board can be sparse and uneven across strikes and maturities. The central research question of this project is whether historical surface dynamics and dependencies between option contracts provide an advantage over classical arbitrage-aware methods when the current option board contains few reliable observations. Using Moscow Exchange options, we study both reconstruction of the current surface from incomplete quotes and short-horizon forecasting of future surfaces. Classical SVI/SSVI-type and smoothing baselines are compared with machine-learning and deep-learning models that use previous market states and cross-contract structure. Evaluation combines controlled quote masking, chronological out-of-sample forecasting, static-arbitrage diagnostics, and robustness analysis across volatility regimes.

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{citekey,
    title={Temporal and Cross-Contract Modeling of Sparse Implied Volatility Surfaces: Arbitrage-Aware Reconstruction and Forecasting},
    author={Name Surname, Name Surname (consultant), Name Surname (advisor)},
    year={2026}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
