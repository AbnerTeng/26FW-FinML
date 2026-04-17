from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .utils import load_data, preprocess, get_array


class EvalMetrics:
    def __init__(self) -> None:
        pass

    def evaluate(self, preds: np.ndarray) -> Dict[str, float]:
        # Implement evaluation logic here
        return {}


class Baselines:
    def __init__(self, data_path: str, look_back: int, future_horizon: int, top_k: int) -> None:
        self.data = load_data(data_path)
        self.look_back = look_back
        self.future_horizon = future_horizon
        self.top_k = top_k
        self.train, _, _ = preprocess(
            self.data,
            ma_windows=[5, 10, 20],
            train_start="2016-01-01",
            val_start="2020-01-01",
            test_start="2021-01-01",
            test_end="2024-12-30",
            add_daily_norm=False
        )
        self.train_feat, self.train_label = get_array(self.train, ["close", "ret"], "cumret_5", False)
        self.dates, self.n_stocks, _ = self.train_feat.shape

    def _calc_momentum(self, return_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Buy top-K past winners (highest cumulative return over look_back window)."""
        momentum_returns = []
        period_ends = []

        for i in range(0, self.dates - self.look_back, self.future_horizon):
            past_cumulative_returns = np.cumsum(return_data[i: i + self.look_back, :], axis=0)[-1]
            top_k_stocks = np.argsort(past_cumulative_returns)[-self.top_k:]  # top-K winners

            future_end = min(i + self.look_back + self.future_horizon, self.dates)
            future_returns = return_data[i + self.look_back: future_end, :]

            momentum_returns.append(future_returns[:, top_k_stocks].mean())
            period_ends.append(future_end)

        return np.array(period_ends), np.cumsum(momentum_returns)

    def _calc_reversal(self, return_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Buy bottom-K past losers (lowest cumulative return over look_back window)."""
        reversal_returns = []
        period_ends = []

        for i in range(0, self.dates - self.look_back, self.future_horizon):
            past_cumulative_returns = np.cumsum(return_data[i: i + self.look_back, :], axis=0)[-1]
            bottom_k_stocks = np.argsort(past_cumulative_returns)[:self.top_k]  # bottom-K losers

            future_end = min(i + self.look_back + self.future_horizon, self.dates)
            future_returns = return_data[i + self.look_back: future_end, :]

            reversal_returns.append(future_returns[:, bottom_k_stocks].mean())
            period_ends.append(future_end)

        return np.array(period_ends), np.cumsum(reversal_returns)

    def build_baseline(self) -> Dict[str, float]:
        return_data = self.train_feat[:, :, 1]  # "ret" is at index 1

        bah_cum = np.cumsum(return_data.mean(axis=1))
        mom_ends, mom_cum = self._calc_momentum(return_data)
        rev_ends, rev_cum = self._calc_reversal(return_data)

        return {
            "buy_and_hold": float(bah_cum[-1]),
            "momentum": float(mom_cum[-1]),
            "reversal": float(rev_cum[-1]),
        }

    def plot_baseline(self, save_path: str = "baseline.pdf") -> None:
        return_data = self.train_feat[:, :, 1]

        bah_cum = np.cumsum(return_data.mean(axis=1))
        mom_ends, mom_cum = self._calc_momentum(return_data)
        rev_ends, rev_cum = self._calc_reversal(return_data)

        days = np.arange(self.dates)

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(14, 6))

        COLORS = {
            "bah": "#2563EB",
            "mom": "#16A34A",
            "rev": "#DC2626",
        }

        # Buy & hold — daily resolution
        ax.plot(days, bah_cum, color=COLORS["bah"], linewidth=2.0,
                label="Buy & Hold (equal weight)", zorder=3)
        ax.fill_between(days, 0, bah_cum,
                        where=bah_cum >= 0, color=COLORS["bah"], alpha=0.08)
        ax.fill_between(days, 0, bah_cum,
                        where=bah_cum < 0, color=COLORS["bah"], alpha=0.08)

        # Momentum — per rebalancing period
        ax.step(mom_ends, mom_cum, where="post", color=COLORS["mom"],
                linewidth=1.8, linestyle="--", label="Momentum (L/S)", zorder=4)
        ax.fill_between(mom_ends, 0, mom_cum, step="post",
                        where=mom_cum >= 0, color=COLORS["mom"], alpha=0.10)
        ax.fill_between(mom_ends, 0, mom_cum, step="post",
                        where=mom_cum < 0, color=COLORS["mom"], alpha=0.10)

        # Reversal — per rebalancing period
        ax.step(rev_ends, rev_cum, where="post", color=COLORS["rev"],
                linewidth=1.8, linestyle="-.", label="Reversal (L/S)", zorder=4)
        ax.fill_between(rev_ends, 0, rev_cum, step="post",
                        where=rev_cum >= 0, color=COLORS["rev"], alpha=0.10)
        ax.fill_between(rev_ends, 0, rev_cum, step="post",
                        where=rev_cum < 0, color=COLORS["rev"], alpha=0.10)

        # Zero baseline
        ax.axhline(0, color="#6B7280", linewidth=0.8, linestyle=":", zorder=2)

        # Final-value annotations
        for label, x, y, color in [
            ("B&H",  days[-1],    bah_cum[-1], COLORS["bah"]),
            ("MOM",  mom_ends[-1], mom_cum[-1], COLORS["mom"]),
            ("REV",  rev_ends[-1], rev_cum[-1], COLORS["rev"]),
        ]:
            ax.annotate(
                f"{label}\n{y:+.3f}",
                xy=(x, y),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=14,
                color=color,
                fontweight="bold",
                va="center",
            )

        ax.set_title("Baseline Strategy Cumulative Returns", fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("Trading Day", fontsize=16)
        ax.set_ylabel("Cumulative Return", fontsize=16)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(frameon=True, fontsize=16, loc="upper left")
        ax.margins(x=0.01)

        fig.tight_layout()
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Saved to {save_path}")
        plt.close(fig)


if __name__ == "__main__":
    baselines = Baselines("./data/ni225_stock.pkl", look_back=20, future_horizon=5, top_k=5)
    print(baselines.build_baseline())
    baselines.plot_baseline("baseline.pdf")
