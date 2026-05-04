# debug_filter_check.py  — run this manually on one file to verify
import matplotlib
import matplotlib.pyplot as plt
from eeg_cnn_lstm.preprocessing.preprocessing import load_edf, filter_raw

matplotlib.use("Agg")  # non-interactive backend for headless server


edf_path = "path/to/one/file.edf"
raw = load_edf(edf_path)

fig_before, ax_before = plt.subplots()
raw.compute_psd(fmax=80).plot(axes=ax_before, show=False)
ax_before.set_title("PSD — Before filtering")
fig_before.savefig("psd_before.png", dpi=150)

filter_raw(raw)

fig_after, ax_after = plt.subplots()
raw.compute_psd(fmax=80).plot(axes=ax_after, show=False)
ax_after.set_title("PSD — After filtering")
fig_after.savefig("psd_after.png", dpi=150)

print("Saved psd_before.png and psd_after.png")
