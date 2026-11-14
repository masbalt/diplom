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

Implied volatility surfaces are used for option pricing, hedging and risk management. On the Moscow Exchange only about one fifth of the listed strikes trade on a given day, so trades reveal the surface only in part.

This work tests whether market history helps to reconstruct and forecast the surface from a small number of current trades, and proposes a method that uses this history. The method takes a starting curve from history (yesterday's exchange settlement curve, or a one-day-ahead forecast), corrects it with a Gaussian process fitted to the visible trades of the current day, with a covariance learned from the previous year of trades, and then removes static arbitrage by linear programming.

The method is compared with a quadratic smile, SVI and SSVI, which use current trades only, with simple methods based on yesterday's curve, and with a neural network trained on the same data. Parameters are estimated on one year of trades in options on USD/RUB futures (Si). Accuracy is measured on the next year (254 days, 21,217 trades): real trades are hidden under twelve scenarios and reconstructed from the remaining ones. With 90% of the day's trades hidden the error is 1.12 volatility points against 1.97 for SSVI; with 10% hidden it is 0.80 against 1.05 for the best current-board method. Forecasts for the following days improve when the curve is carried over in normalised moneyness and corrected by the latest trades: trades lead the exchange curve. A neural network with the same starting curve reaches similar accuracy. The main result holds on a 29-day control period whose data were downloaded after the methods were fixed.

Known limits: links between option series did not improve accuracy; one erroneous exchange curve in the control period noticeably hurt every method that relies on yesterday's curve (a curve check added afterwards removes that error, but it still needs confirmation on new data); the arbitrage repair costs accuracy when very few trades are visible; on options on RTS futures the forecast and the one-day-ahead starting curve gave no gain.

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{sitnikov2026ivs,
    title={Temporal and Cross-Contract Modeling of Sparse Implied Volatility Surfaces: Arbitrage-Aware Reconstruction and Forecasting},
    author={Aleksandr Sitnikov, S. I. Gurov (advisor)},
    year={2026}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
