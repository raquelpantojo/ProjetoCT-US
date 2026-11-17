"""
Diagnostico_etapas.py
=====================
Abre o pipeline TC -> US (modelo de Kutter, Shams & Navab 2009) etapa por etapa em UMA fatia e
mede quanto cada termo da Eq. 6 pesa na imagem final:

    I_us = ( G_s1 * I_r  +  alpha * G_s2 * alpha_T ) * I_s
           \\_reflexao_/    \\____base do speckle____/

Use para entender por que a imagem fica "com muito espalhamento" e para testar parametros.
Requer CT_para_US_mamas.py e TransformCTUS_kutter.py na mesma pasta.
"""
import os
import sys
import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import hilbert
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import CT_para_US_mamas as C                                   # noqa: E402
from TransformCTUS_kutter import SimuladorKutter, Z_AGUA, Z_AR  # noqa: E402

# ----------------------------- PARAMETROS -----------------------------
PASTA = os.path.dirname(os.path.abspath(__file__))
CAMINHO_SERIE = os.path.join(PASTA, "2-CTAC-14691")
PASTA_PET = os.path.join(PASTA, "3-PET AC 3DWB-52072")      # None ou inexistente = sem inclusao
MAMA = "mama_esquerda"
FATIA = 54                     # fatia da TC (None = maior area da mama)
PARAMETROS_SIMULADOR = dict(f0_mhz=5.0)   # mesmos padroes do simulador
NORMALIZAR_PICO_G1 = None      # None = usa o simulador; False = versao antiga (pico dividido por ~36)
ARQUIVO_FIGURA = os.path.join(PASTA, "diagnostico_etapas.png")
# ----------------------------------------------------------------------


def etapas_fatia(sim, hu_slice, corpo_slice, rn_slice, m_slice, coords, spacing, normalizar_pico_g1=None,
                 suavizar_hu_z_mm=None):
    """Mesmas contas de SimuladorKutter.simular_fatia, mas devolvendo todos os intermediarios.
    suavizar_hu_z_mm > 0 suaviza a TC so para calcular Z (as reflexoes deixam de vir do ruido)."""
    dx, dy, _ = spacing
    E = {}
    if normalizar_pico_g1 is None:
        normalizar_pico_g1 = sim.normalizar_pico_reflexao
    if suavizar_hu_z_mm is None:
        suavizar_hu_z_mm = sim.suavizacao_ct_z_mm
    hu_slice = np.where(corpo_slice, hu_slice, 0.0).astype(np.float32)
    if suavizar_hu_z_mm > 0:                      # mesma suavizacao normalizada do simulador
        sg = suavizar_hu_z_mm / dx
        peso = ndimage.gaussian_filter(corpo_slice.astype(np.float32), sg)
        hu_z = np.where(corpo_slice, ndimage.gaussian_filter(hu_slice * corpo_slice, sg) / np.maximum(peso, 1e-3), 0.0)
    else:
        hu_z = hu_slice
    hu = ndimage.map_coordinates(hu_slice, coords, order=1, mode="nearest")
    corpo = ndimage.map_coordinates(ndimage.gaussian_filter(corpo_slice.astype(np.float32), 1.0),
                                    coords, order=1, mode="constant") > 0.5
    w = corpo.astype(np.float32)
    rn = ndimage.map_coordinates(rn_slice, coords, order=1, mode="nearest")
    m = ndimage.map_coordinates(m_slice, coords, order=1, mode="constant") * w
    hu = hu * w
    E["HU"], E["m"], E["rn"] = hu, m, rn
    huz = ndimage.map_coordinates(hu_z, coords, order=1, mode="nearest") * w

    Z = Z_AGUA * (huz + 1000.0) / 1000.0
    Z = np.where(huz < sim.limiar_ar_hu, Z_AR, Z)
    Z = np.where(huz > sim.limiar_osso_hu, sim.z_osso, Z)
    E["Z"] = np.maximum(Z, Z_AR)

    h = max(int(round(dy / (2.0 * sim.dz_fino))), 1)
    Zp = np.pad(E["Z"], ((0, h), (0, 0)), mode="edge")[h:]
    Zm = np.pad(E["Z"], ((h, 0), (0, 0)), mode="edge")[:-h]
    Rd = ((Zp - Zm) / (Zp + Zm)) ** 2
    pico = np.zeros_like(Rd, dtype=bool)
    pico[1:-1] = (Rd[1:-1] >= Rd[:-2]) & (Rd[1:-1] > Rd[2:]) & (Rd[1:-1] > 1e-6)
    E["aR"] = np.where(pico, Rd, 0.0).astype(np.float32)

    Ii = np.cumprod(1.0 - E["aR"], axis=0)
    E["Ii"] = np.vstack([np.ones((1, Ii.shape[1]), Ii.dtype), Ii[:-1]])

    Ir = E["aR"] * E["Ii"] ** 2 * rn
    E["Ir"] = ndimage.convolve1d(Ir, sim._janela_lateral(sim.na_reflexao), axis=1, mode="constant")

    rf = sim._espalhamento_rf(m)
    Is = np.abs(hilbert(rf, axis=0)) ** 2 / sim._normalizacao_is()
    E["Is"] = ndimage.convolve1d(Is, sim._janela_lateral(sim.na_espalhamento), axis=1, mode="nearest")

    s1 = (sim.sigma_reflexao_mm / sim.dz_fino, sim.sigma_reflexao_mm / sim.dx_fino)
    s2 = (sim.sigma_transmissao_mm / sim.dz_fino, sim.sigma_transmissao_mm / sim.dx_fino)
    G1 = ndimage.gaussian_filter(E["Ir"], s1, mode="nearest")
    E["fator_G1"] = np.sqrt(2 * np.pi) * s1[0]          # perda de pico de uma interface de 1 amostra
    if normalizar_pico_g1:
        G1 = G1 * E["fator_G1"]
    E["G1_Ir"] = G1
    E["base"] = sim.alpha_mistura * ndimage.gaussian_filter(E["Ii"], s2, mode="nearest")
    E["Ius"] = (E["G1_Ir"] + E["base"]) * E["Is"]
    E["s1_amostras"] = s1
    return E


def db(x, ref):
    return 10 * np.log10(np.maximum(x / ref, 1e-12))


if __name__ == "__main__":
    img, vol, tags = C.carregar_serie(CAMINHO_SERIE)
    sp = img.GetSpacing()
    corpo = SimuladorKutter.mascara_corpo(vol)
    sep = C.separar_mamas(vol, corpo, sp)
    incl = None
    if PASTA_PET and os.path.isdir(PASTA_PET):
        suv, _ = C.carregar_suv(PASTA_PET, img)
        inc, _ = C.segmentar_inclusoes(suv, sep["mascaras"], corpo, sp)
        incl = inc[MAMA]
        if incl is not None:
            sep["mascaras"][MAMA] |= ndimage.binary_dilation(incl, np.ones((1, 3, 3), bool), iterations=2) & corpo
    masc = sep["mascaras"][MAMA]
    k0, k1, r0, r1, c0, c1 = C.caixa_da_mama(masc, sp, meio=sep["meio"], lado=MAMA)
    vol_c, corpo_c, masc_c = vol[k0:k1, r0:r1, c0:c1], corpo[k0:k1, r0:r1, c0:c1], masc[k0:k1, r0:r1, c0:c1]
    k = (FATIA if FATIA is not None else k0 + int(np.argmax(masc_c.sum(axis=(1, 2))))) - k0

    sim = SimuladorKutter(**PARAMETROS_SIMULADOR)
    rn, m, _ = sim.preprocessar_volume(vol_c, corpo_c, sp, roi_3d=masc_c.astype(np.float32))
    if incl is not None:
        m = np.where(incl[k0:k1, r0:r1, c0:c1], np.float32(0.1), m)
    coords, prof, xs = sim.coordenadas_regiao(sp, 0.0, vol_c.shape[2] * sp[0], 0.0, vol_c.shape[1] * sp[1])
    E = etapas_fatia(sim, vol_c[k], corpo_c[k], rn[k], m[k], coords, sp, NORMALIZAR_PICO_G1)
    dentro = ndimage.map_coordinates(masc_c[k].astype(np.float32), coords, order=1) > 0.5

    # ---------------- numeros ----------------
    aR_int = E["aR"][dentro & (E["aR"] > 0)]
    G1_int = E["G1_Ir"][dentro]
    base = E["base"][dentro]
    print(f"Fatia {k0 + k}, {MAMA}")
    print(f"  alpha_R nas interfaces (dentro da mama): mediana {np.median(aR_int):.2e}, p99 {np.percentile(aR_int, 99):.2e}"
          f"  (pele/agua: {E['aR'].max():.2e})")
    print(f"  G_s1 tem desvio de {E['s1_amostras'][0]:.1f} amostras axiais: o pico de uma interface cai "
          f"{E['fator_G1']:.0f}x ({10 * np.log10(E['fator_G1']):.1f} dB)"
          + ("  [compensado no simulador]" if (NORMALIZAR_PICO_G1 if NORMALIZAR_PICO_G1 is not None else sim.normalizar_pico_reflexao) else ""))
    print(f"  termo de reflexao G1*I_r: p50 {np.median(G1_int):.2e}, p99 {np.percentile(G1_int, 99):.2e}")
    print(f"  base do speckle alpha*G2*aT: p50 {np.median(base):.2e}  (alpha = {sim.alpha_mistura})")
    razao = np.percentile(G1_int, 99) / np.median(base)
    print(f"  reflexoes mais fortes / base do speckle = {razao:.2f}  ({10 * np.log10(razao):+.1f} dB)")
    print(f"  pixels da mama em que a reflexao domina: {100 * np.mean(G1_int > base):.1f}%")
    print(f"  contraste glandula x gordura no mapa m: p90/p10 = {np.percentile(E['m'][dentro], 90) / np.percentile(E['m'][dentro], 10):.1f}x")

    # ---------------- figura ----------------
    ext = [xs[0], xs[-1], prof[-1], prof[0]]
    paineis = [
        ("1. HU (TC reamostrada)", E["HU"], dict(cmap="bone", vmin=-200, vmax=200)),
        ("2. Z (MRayl)", E["Z"] / 1e6, dict(cmap="viridis", vmin=1.3, vmax=1.6)),
        ("3. alpha_R (log10)", np.log10(E["aR"] + 1e-7), dict(cmap="magma", vmin=-5, vmax=-1)),
        ("4. I_i (transmissao)", E["Ii"], dict(cmap="gray", vmin=0.9, vmax=1.0)),
        ("5. |r.n| (Lambert)", E["rn"], dict(cmap="gray", vmin=0, vmax=1)),
        ("6. mapa de espalhamento m", E["m"], dict(cmap="gray", vmin=0, vmax=1)),
        ("7. I_s (speckle)", db(E["Is"], np.percentile(E["Is"][dentro], 99)), dict(cmap="gray", vmin=-40, vmax=0)),
        ("8. G1*I_r (reflexao), dB", db(E["G1_Ir"] + 1e-12, np.median(base)), dict(cmap="inferno", vmin=-30, vmax=10)),
        ("9. I_us final (Eq. 6)", db(E["Ius"], np.percentile(E["Ius"][dentro], 99.5)), dict(cmap="gray", vmin=-sim.faixa_dinamica_db, vmax=0)),
    ]
    fig, ax = plt.subplots(3, 3, figsize=(16, 13))
    for a, (t, im, kw) in zip(ax.ravel(), paineis):
        h = a.imshow(np.where(dentro | (t.startswith("1")), im, np.nan), extent=ext, aspect="equal", **kw)
        a.set_title(t, fontsize=11); plt.colorbar(h, ax=a, fraction=0.046)
    ax[2, 1].set_xlabel(f"Reflexao relativa a base do speckle (0 dB = alpha*G2*aT)")
    plt.suptitle(f"Etapas do modelo de Kutter - fatia {k0 + k}, {MAMA.replace('_', ' ')}"
                 f"  |  reflexoes p99 / base = {10 * np.log10(razao):+.1f} dB", fontsize=13)
    plt.tight_layout()
    plt.savefig(ARQUIVO_FIGURA, dpi=75)
    print(f"Figura: {ARQUIVO_FIGURA}")
    plt.show()
