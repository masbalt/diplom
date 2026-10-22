# Arbitrage-Aware Reconstruction and Short-Horizon Forecasting of Implied Volatility Surfaces from Sparse Option Trades Using Historical Dynamics and Cross-Contract Dependencies

[![License](https://badgen.net/github/license/masbalt/diplom?color=green)](https://github.com/masbalt/diplom/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/masbalt/diplom)](https://github.com/masbalt/diplom/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/masbalt/diplom.svg?color=0088ff)](https://github.com/masbalt/diplom/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/masbalt/diplom.svg?color=7f29d6)](https://github.com/masbalt/diplom/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Aleksandr Sitnikov </td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Name Surname, PhD/DSc </td>
    </tr>
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td> S.I.Gurov, PhD/DSc </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/main.pdf)

## Abstract

Implied volatility surfaces are widely used in option pricing, hedging, and risk management. However, on less liquid markets, option trades are unevenly distributed across strike prices and maturities, making reliable surface reconstruction difficult.

This study aims to develop and evaluate an arbitrage-aware method for reconstructing implied volatility surfaces from sparse market transactions using historical market information and dependencies between option contracts with different strikes and maturities. The Moscow Exchange options market, primarily options on USD/RUB futures (Si), serves as the empirical setting.

The proposed approach will combine current transaction data with historical volatility curves and cross-contract information. Its reconstruction accuracy and financial consistency will be evaluated against classical methods, including SVI/SSVI, arbitrage-aware smoothing techniques, and historical baselines.

Preliminary experiments indicate that historical information can substantially improve reconstruction accuracy when the number of current observations is particularly small. The study will further examine how this advantage depends on data sparsity, assess static-arbitrage violations, and investigate the possibility of short-horizon surface forecasting.

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
