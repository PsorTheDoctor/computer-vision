import cv2
import numpy as np
from scipy import stats


def get_luminance(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return L


def compute_gradient_features(L):
    Gx = cv2.Sobel(L, cv2.CV_64F, dx=1, dy=0, ksize=3)
    Gy = cv2.Sobel(L, cv2.CV_64F, dx=0, dy=1, ksize=3)

    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    M = np.stack((Gx_flat, Gy_flat), axis=1)

    n = M.shape[0]
    C = (1 / n) * np.dot(M.T, M)  # covariance matrix

    eigvals, _ = np.linalg.eigh(C)
    eigvals = eigvals[::-1]
    l1, l2 = eigvals[:2]
    l1 = max(l1, 1e-10)
    l2 = max(l2, 1e-10)
    rho = l1 / l2  # anisotropy ratio
    kappa = ((l1 - l2) / (l1 + l2)) ** 2  # coherence
    energy = l1 + l2  # gradient energy
    return rho, kappa, energy, l1, l2, Gx, Gy


def compute_frequency_features(L):
    h, w = L.shape

    # FFT (Fast Fourier Transform)
    F = np.fft.fft2(L)
    F_shift = np.fft.fftshift(F)
    magnitude_spectrum = np.abs(F_shift)
    power_spectrum = magnitude_spectrum ** 2

    # Radial profile
    x_center, y_center = w // 2, h // 2
    y, x = np.ogrid[-x_center: h - x_center, -y_center: w - y_center]
    r = np.sqrt(x * x + y * y)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / np.maximum(nr, 1)

    # Spectral slope (beta)
    r_axis = np.arange(len(radial_profile))
    mask = (r_axis > 0) & (r_axis < min(h, w) // 2)
    log_r = np.log10(r_axis[mask])
    log_S = np.log10(radial_profile[mask] + 1e-10)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_S)
    beta = -slope

    # High frequency ratio (eta)
    cutoff = min(h, w) / 4
    total_energy = np.sum(radial_profile[1:])
    high_freq_energy = np.sum(radial_profile[r_axis > cutoff])
    if total_energy > 0:
        eta = high_freq_energy / total_energy
    else:
        eta = 0.0

    return beta, eta, radial_profile, log_r, log_S, slope, intercept, power_spectrum


img = cv2.imread('../data/adam.jpg')
L = get_luminance(img)
rho = compute_gradient_features(L)[0]
beta, eta = compute_frequency_features(L)[:2]

is_physics_natural = 1.6 <= beta <= 2.6
is_anisotropic = rho > 1.15
is_compressed = eta < 0.01
is_high_freq_noise = eta > 0.25

score = 0
reasons = []
if is_physics_natural:
    score += 1
else:
    print("Spectral Slope (Beta) deviates from natural light physics.")
if is_anisotropic:
    score += 1
else:
    print("Gradient field is Isotropic (lacks directional coherence).")
if is_high_freq_noise:
    score -= 10
    print("High-frequency energy is excessive (Diffusion artifacts).")

if is_compressed and is_physics_natural:
    print("REAL (Compressed)")
    print("Light physics (Beta) is correct. The lack of fine details is likely due to compression, not AI generation.")
elif score == 2:
    print("LIKELY REAL")
    print("Image passes all physical and structural tests.")
elif score == 1:
    print("UNCERTAIN / HYBRID")
    print("Image is ambiguous: passes some but not all natural image tests.")
else:
    print("LIKELY SYNTHETIC (AI)")
    print("Image fails key natural image tests. " + (" ".join(reasons) if reasons else ""))
