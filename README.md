# Arbitrage-Aware Reconstruction and Short-Horizon Forecasting of Implied Volatility Surfaces in Sparse Option Markets

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

Implied volatility surfaces are a central input for option valuation, hedging, and risk management because they summarize market-implied uncertainty across strike prices and expiration dates. On less liquid option markets, however, quotes are unevenly distributed across strikes and maturities, so the observable surface is sparse, noisy, and may contain violations of static no-arbitrage conditions. This study investigates two connected problems: reconstruction of a financially consistent implied volatility surface from incomplete option quotes and short-horizon forecasting of its subsequent evolution, using the Moscow Exchange option market as the main empirical setting. Classical parametric approaches are compared with machine-learning and deep-learning methods under controlled quote sparsity, chronological out-of-sample testing, volatility-regime changes, and explicit static-arbitrage diagnostics. The goal is to determine whether flexible data-driven models can improve reconstruction and forecasting accuracy on a relatively illiquid option market without sacrificing the financial consistency required for derivative pricing and risk management.

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{citekey,
    title={Arbitrage-Aware Reconstruction and Short-Horizon Forecasting of Implied Volatility Surfaces in Sparse Option Markets},
    author={Name Surname, Name Surname (consultant), Name Surname (advisor)},
    year={2026}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
