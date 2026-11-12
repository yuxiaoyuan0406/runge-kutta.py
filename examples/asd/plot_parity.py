'''
Overlay the MATLAB reference and the Python `asd_fn` result on one figure.

Reads the data committed for the test suite
(`util/tests/asd_test_data/`): the input series and the MATLAB output for
`fl=10 Hz, fh=300 Hz @ 128 kHz`. The top panel draws both ASD curves (they
coincide); the bottom panel plots the relative PSD difference, which is the
part that actually shows they agree.

Usage (from the repository root):

    python examples/asd/plot_parity.py -o ../figures/asd_parity.png
    python examples/asd/plot_parity.py --show

The optional `--show` flag displays the figure instead of just saving it.
'''
import argparse
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt

# Same convention as the other examples: run from the repository root.
sys.path.append(".")
from util.asd import asd_fn, plot_asd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "util" / "tests" / "asd_test_data"
CASE = "asd_test_data_128kHz_10_300"
REF = "matlab_ref_128kHz_10_300"


def load_case(data_dir=DATA_DIR):
    '''Load the input series and the MATLAB reference, plus their metadata.'''
    ref = np.load(Path(data_dir) / f"{REF}.npz")
    with h5py.File(Path(data_dir) / f"{CASE}.mat", "r") as f:
        x = np.array(f["chopout4"]).ravel()
    meta = {
        "fs": float(ref["fs"]),
        "fl": float(ref["fl"]),
        "fh": float(ref["fh"]),
        "nfft": int(ref["nfft"]),
        "n": int(ref["n"]),
    }
    return x, ref["freq"], ref["psd"], float(ref["nf_db"]), meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("-o", "--output", default="figures/asd_parity.png",
                        help="where to save the PNG (default: figures/asd_parity.png)")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help="folder holding the .mat input and the .npz reference")
    parser.add_argument("--show", action="store_true", help="display the figure as well")
    args = parser.parse_args(argv)

    x, freq_m, psd_m, nf_m, meta = load_case(args.data_dir)
    asd_m = np.sqrt(psd_m)

    freq, asd, nf = asd_fn(x, meta["fl"], meta["fh"], fs=meta["fs"], nfft=meta["nfft"])

    psd = asd ** 2
    rel = np.abs(psd - psd_m) / np.abs(psd_m)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9.0, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    ax_top.set_xscale("log")

    n_before = len(ax_top.get_lines())
    plot_asd(freq_m, asd_m, nf_m, ax=ax_top, label="MATLAB reference (R2025a)",
             band=(meta["fl"], meta["fh"]))
    matlab_curve = ax_top.get_lines()[n_before]
    matlab_curve.set_linewidth(4.0)
    matlab_curve.set_alpha(0.35)
    matlab_curve.set_color("C0")

    n_before = len(ax_top.get_lines())
    plot_asd(freq, asd, nf, ax=ax_top, label="Python asd_fn",
             band=(meta["fl"], meta["fh"]))
    python_curve = ax_top.get_lines()[n_before]
    python_curve.set_linewidth(1.0)
    python_curve.set_color("k")
    python_curve.set_linestyle("--")

    ax_top.axvspan(meta["fl"], meta["fh"], color="C2", alpha=0.08, zorder=0)
    # Rebuild the legend now that the two curves have been restyled, otherwise
    # the entries keep the colours they had when plot_asd created the legend.
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.set_title(
        f"MATLAB vs Python ASD — {CASE}\n"
        f"N={meta['n']} samples, fs={meta['fs']:g} Hz, nfft={meta['nfft']}, "
        f"noise-floor band {meta['fl']:g}-{meta['fh']:g} Hz",
        fontsize=11,
    )
    ax_top.text(
        0.015, 0.05,
        f"max |ΔPSD|/PSD = {rel.max():.2e}\n"
        f"nf: MATLAB {nf_m:.9f} dB, Python {nf:.9f} dB (Δ = {nf - nf_m:+.1e} dB)",
        transform=ax_top.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax_bot.semilogy(freq_m, np.maximum(rel, 1e-18), lw=0.8, color="C3",
                    label="|ΔPSD| / PSD")
    ax_bot.axhline(1e-9, ls=":", color="k", lw=1.0, label="test tolerance (rtol=1e-9)")
    ax_bot.set_ylabel("relative diff.")
    ax_bot.set_xlabel("Frequency [Hz]")
    ax_bot.grid(True, which="both", alpha=0.3)
    ax_bot.legend(loc="upper right", fontsize=8)
    left, right = ax_top.get_xlim()
    ax_bot.set_xlim(left, right)

    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"saved {args.output}")
    print(f"max |ΔPSD|/PSD = {rel.max():.3e}, Δnf = {nf - nf_m:+.3e} dB")
    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
