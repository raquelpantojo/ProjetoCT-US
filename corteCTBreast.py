"""
CT_para_US_mamas.py
===================
Pipeline completo TC -> ultrassom simulado para exames de mama (QIN-BREAST, decubito ventral):

  1. Le a serie DICOM da TC e reorienta para LPS (linha 0 = anterior, coluna -> esquerda do paciente).
  2. Segmenta o corpo (o suporte de mama da mesa e o ar externo ficam de fora).
  3. Separa automaticamente a MAMA DIREITA e a MAMA ESQUERDA:
       - linha media = sulco intermamario (ponto mais posterior do perfil anterior do corpo entre
         as duas mamas);
       - limite posterior = pele na linha media, fatia a fatia, + OFFSET_CORTE_MM (inclui a base
         da mama);
       - so entram as fatias em que a mama se projeta pelo menos PROTRUSAO_MIN_MM.
  4. Recorta a TC de cada mama (o "corte da imagem real").
  5. Simula o ultrassom de cada mama pelo modelo de Kutter, Shams & Navab (2009), usando o
     SimuladorKutter de TransformCTUS_kutter.py (Secao 7 do documento Estudo_CT_para_US.docx):
       Z linear no HU (ar/osso segmentados) -> alpha_R, alpha_T -> I_i recursivo -> reflexao
       lambertiana com abertura ativa (Hann) -> espalhamento (Poisson + PSF + envelope, alongado)
       -> I_us = (G_s1 * I_r + alpha G_s2 * alpha_T) I_s -> compressao log [Eq. 7].
     Phantom de US: a mama inteira e simulada como se fosse varrida de cima, com o feixe no
     sentido anterior -> posterior (do mamilo para a parede toracica), sem posicao de
     transdutor e sem limite de profundidade (a imagem cobre todo o recorte da mama).
     A fisica e calculada com todo o tecido dentro do recorte (sem interfaces artificiais no
     corte); so no fim os pixels fora da mama sao zerados.
  6. Grava, para cada mama, novas series DICOM:
       saida/<mama>/US/  -> ultrassom simulado (8 bits, 0 = -faixa dinamica, 255 = 0 dB)
       saida/<mama>/CT/  -> TC recortada (HU, int16; fora da mama = -1000 se MASCARAR_CT)
     As series ficam no mesmo estudo e no mesmo Frame of Reference da TC original, com posicao
     e orientacao (ImagePositionPatient / ImageOrientationPatient) corretas: em visualizadores
     como o 3D Slicer, US e TC se sobrepoem.

Convencao de lados: em LPS a coluna cresce para a ESQUERDA do paciente. Na tela (convencao
radiologica) a mama direita aparece a esquerda da imagem.

Inclusoes (lesoes): se a pasta do PET do mesmo estudo PET/CT for informada, o FDG localiza o
tumor (SUVmax >= SUV_MINIMO; segmentacao a 40% do SUVmax). A lesao e somada a mascara da mama
(mesmo alem do limite posterior) e recebe ecogenicidade propria (hipoecoica) no simulador;
uma serie DICOM extra (INCLUSAO) guarda o mapa de rotulos.

Requisitos: pip install SimpleITK scipy numpy matplotlib pydicom
TransformCTUS_kutter.py precisa estar na mesma pasta deste script.
"""

import os
import sys
import time
import datetime
import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from TransformCTUS_kutter import SimuladorKutter  # noqa: E402

LADOS = ("mama_direita", "mama_esquerda")


# ====================================================================== #
# 1. Leitura
# ====================================================================== #
def carregar_serie(caminho_pasta):
    """Devolve (imagem SimpleITK em LPS, volume HU [Z,Y,X], tags DICOM da 1a fatia)."""
    reader = sitk.ImageSeriesReader()
    nomes = reader.GetGDCMSeriesFileNames(caminho_pasta)
    if not nomes:
        raise FileNotFoundError(f"Nenhum arquivo DICOM encontrado em: {caminho_pasta}")
    reader.SetFileNames(nomes)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    img = reader.Execute()
    tags = {k: reader.GetMetaData(0, k) for k in reader.GetMetaDataKeys(0)}
    img = sitk.DICOMOrient(img, "LPS")
    vol = np.maximum(sitk.GetArrayFromImage(img).astype(np.float32), -1000.0)  # -3024 (fora do FOV) -> ar
    return img, vol, tags


# ====================================================================== #
# 2-3. Separacao das mamas
# ====================================================================== #
def perfil_anterior(corpo):
    """Primeira linha do corpo em cada coluna, por fatia: f[k, x] (inf se nao ha corpo)."""
    tem = corpo.any(axis=1)                                   # (Z, X)
    primeira = np.argmax(corpo, axis=1).astype(np.float32)    # (Z, X)
    primeira[~tem] = np.inf
    return primeira


def separar_mamas(vol, corpo, spacing, offset_corte_mm=10.0, protrusao_min_mm=10.0,
                  meio_coluna=None, distancia_min_mm=40.0):
    """
    Separa mama direita e esquerda.
    Devolve dict com: meio (coluna), corte (linha limite por fatia), mascaras {lado: bool [Z,Y,X]},
    fatias {lado: array de indices}, mamilos {lado: coluna do ponto mais anterior}.
    """
    dx, dy, _ = spacing
    nz, ny, nx = corpo.shape
    f = perfil_anterior(corpo)
    F = np.where(np.isfinite(f), f, ny).min(axis=0)           # perfil anterior global (min nas fatias)
    F = ndimage.gaussian_filter1d(F, 3.0)
    cols_corpo = np.where(corpo.any(axis=(0, 1)))[0]
    centro = int(round(cols_corpo.mean()))

    # mamilos: ponto mais anterior de cada lado do centro do corpo
    dmin = int(round(distancia_min_mm / 2.0 / dx))
    dir_ = np.arange(cols_corpo[0], max(centro - dmin, cols_corpo[0] + 1))
    esq_ = np.arange(min(centro + dmin, cols_corpo[-1] - 1), cols_corpo[-1] + 1)
    mamilo_dir = int(dir_[np.argmin(F[dir_])])
    mamilo_esq = int(esq_[np.argmin(F[esq_])])

    # linha media = sulco intermamario (mais posterior entre os dois mamilos)
    if meio_coluna is None:
        entre = np.arange(mamilo_dir, mamilo_esq + 1)
        meio = int(entre[np.argmax(F[entre])])
    else:
        meio = int(meio_coluna)

    # limite posterior, fatia a fatia: pele na linha media + offset
    off = offset_corte_mm / dy
    pele_meio = f[:, meio]
    corte = pele_meio + off                                   # inf onde nao ha corpo na linha media

    linhas = np.arange(ny)[None, :, None]
    colunas = np.arange(nx)[None, None, :]
    anterior = linhas < corte[:, None, None]
    lado_col = {"mama_direita": colunas < meio, "mama_esquerda": colunas >= meio}

    mascaras, fatias = {}, {}
    for lado in LADOS:
        m = corpo & anterior & lado_col[lado]
        # protrusao da mama em cada fatia: pele na linha media - pele mais anterior do lado
        cols = np.where(lado_col[lado][0, 0])[0]
        prot_mm = (pele_meio - f[:, cols].min(axis=1)) * dy
        ok = np.isfinite(prot_mm) & (prot_mm >= protrusao_min_mm)
        m[~ok] = False
        # mantem so o maior componente 3D (remove pedacos soltos)
        rot, n = ndimage.label(m)
        if n > 1:
            tam = ndimage.sum_labels(np.ones_like(rot), rot, index=np.arange(1, n + 1))
            m = rot == (np.argmax(tam) + 1)
        mascaras[lado] = m
        fatias[lado] = np.where(m.any(axis=(1, 2)))[0]
        if fatias[lado].size == 0:
            raise RuntimeError(f"Nao foi possivel encontrar a {lado.replace('_', ' ')}. "
                               "Ajuste PROTRUSAO_MIN_MM ou MEIO_COLUNA.")
    return dict(meio=meio, meio_lado={l: meio for l in LADOS}, corte=corte, mascaras=mascaras, fatias=fatias,
                mamilos={"mama_direita": mamilo_dir, "mama_esquerda": mamilo_esq}, perfil=F)


def fatia_referencia(corpo):
    """Fatia em que o corpo chega mais a frente (em geral a do mamilo da mama mais pendente)."""
    f = perfil_anterior(corpo)
    return int(np.argmin(np.where(np.isfinite(f), f, np.inf).min(axis=1)))


def _verificar_backend_interativo():
    b = plt.get_backend().lower()
    if b == "agg" or "inline" in b:
        raise RuntimeError(
            f"O matplotlib esta com backend '{plt.get_backend()}', que nao abre janela para desenhar as retas.\n"
            "   - Spyder: Ferramentas > Preferencias > IPython console > Graficos > Backend: Automatico (reinicie o console)\n"
            "   - Jupyter: rode '%matplotlib qt' antes\n"
            "   - Ou rode pelo terminal: python CT_para_US_mamas.py\n"
            "   - Ou use MODO_CORTE = 'manual' com LINHAS_CORTE preenchido.")


def _esperar(fig, estado):
    """Bloqueia ate o usuario apertar Enter (ou fechar a janela)."""
    fig.canvas.mpl_connect("close_event", lambda e: estado.update(fechada=True))
    plt.show(block=False)
    while not estado.get("ok") and not estado.get("fechada"):
        plt.pause(0.05)


def _desenhar_retas_axiais(vol, corpo, spacing, k_ref, lado, titulo, retas_anteriores):
    """
    Janela do corte axial. O usuario DESENHA (clica, arrasta e solta):
      reta 1 (amarela): a reta VERTICAL do limite medial - so a posicao x importa, ela fica sempre vertical;
      reta 2 (laranja): a reta de CORTE, inclinada como quiser (do inicio ao fim do arraste).
    A regiao que vai entrar no recorte aparece em vermelho. R refaz, Enter confirma.
    """
    dx, dy, _ = spacing
    ny, nx = vol.shape[1:]
    ys, xs = np.where(corpo[k_ref])
    X = (np.arange(nx) + 0.5) * dx
    Y = (np.arange(ny) + 0.5) * dy
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(vol[k_ref], cmap="bone", vmin=-200, vmax=300, extent=[0, nx * dx, ny * dy, 0])
    for L in retas_anteriores.values():                      # retas da outra mama, para referencia
        (x1, y1), (x2, y2) = L["linha_mm"]
        ax.axvline(L["x_medial_mm"], color="yellow", ls=":", lw=1, alpha=0.6)
        ax.axline((x1, y1), (x2, y2), color="orange", ls=":", lw=1, alpha=0.6)
    ax.set_xlim(max(xs.min() * dx - 20, 0), min(xs.max() * dx + 20, nx * dx))
    ax.set_ylim(min(ys.min() * dy + 0.55 * (ys.max() - ys.min()) * dy, ny * dy), max(ys.min() * dy - 20, 0))
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_autoscale_on(False)

    vert = ax.axvline(np.nan, color="yellow", lw=2)
    seg, = ax.plot([], [], color="orange", lw=2.5)
    ext, = ax.plot([], [], color="orange", lw=1, ls="--")
    over = ax.imshow(np.zeros((ny, nx, 4)), extent=[0, nx * dx, ny * dy, 0])
    est = dict(x_medial=None, p1=None, p2=None, arrastando=False, ok=False)
    instr = {0: "Desenhe a reta 1 (VERTICAL, amarela): clique e arraste sobre o limite medial",
             1: "Desenhe a reta 2 (CORTE, laranja): clique no inicio, arraste ate o fim e solte (pode ser inclinada)",
             2: "Vermelho = o que entra no recorte.   ENTER confirma   |   R refaz as duas retas"}

    def etapa():
        return 0 if est["x_medial"] is None else (1 if est["p2"] is None else 2)

    def atualizar():
        e = etapa()
        ax.set_title(f"{titulo} - fatia {k_ref}\n{instr[e]}", fontsize=10,
                     color="darkgreen" if e == 2 else "darkred")
        vert.set_xdata([est["x_medial"]] * 2 if est["x_medial"] is not None else [np.nan] * 2)
        if est["p1"] is not None and est["p2"] is not None:
            (x1, y1), (x2, y2) = est["p1"], est["p2"]
            seg.set_data([x1, x2], [y1, y2])
            if abs(x2 - x1) > 1e-6:
                xa = np.array(ax.get_xlim())
                ext.set_data(xa, y1 + (xa - x1) * (y2 - y1) / (x2 - x1))
        else:
            seg.set_data([], []); ext.set_data([], [])
        rgba = np.zeros((ny, nx, 4))
        if e == 2 and not est["arrastando"] and abs(est["p2"][0] - est["p1"][0]) > 1e-6:
            (x1, y1), (x2, y2) = est["p1"], est["p2"]
            y_lin = y1 + (X - x1) * (y2 - y1) / (x2 - x1)
            lado_col = X < est["x_medial"] if lado == "mama_direita" else X >= est["x_medial"]
            reg = corpo[k_ref] & (Y[:, None] < y_lin[None, :]) & lado_col[None, :]
            rgba[reg] = (1, 0, 0, 0.30)
        over.set_data(rgba)
        fig.canvas.draw_idle()

    def press(ev):
        if ev.inaxes is not ax or ev.button != 1 or ev.xdata is None:
            return
        e = etapa()
        if e == 0:
            est["x_medial"] = ev.xdata
        elif e == 1:
            est["p1"] = est["p2"] = (ev.xdata, ev.ydata)
        else:
            return
        est["arrastando"] = True
        atualizar()

    def move(ev):
        if not est["arrastando"] or ev.inaxes is not ax or ev.xdata is None:
            return
        if est["p1"] is None:
            est["x_medial"] = ev.xdata
        else:
            est["p2"] = (ev.xdata, ev.ydata)
        atualizar()

    def release(ev):
        if not est["arrastando"]:
            return
        est["arrastando"] = False
        if est["p1"] is not None:
            if ev.inaxes is ax and ev.xdata is not None:
                est["p2"] = (ev.xdata, ev.ydata)
            if abs(est["p2"][0] - est["p1"][0]) < 5 * dx:      # arraste curto/vertical demais: refazer reta 2
                est["p1"] = est["p2"] = None
                print("   (a reta de corte precisa ser arrastada na horizontal/inclinada - desenhe de novo)")
        atualizar()

    def key(ev):
        if ev.key in ("r", "R"):
            est.update(x_medial=None, p1=None, p2=None, arrastando=False)
            atualizar()
        elif ev.key == "enter" and etapa() == 2:
            est["ok"] = True

    for nome, f in (("button_press_event", press), ("motion_notify_event", move),
                    ("button_release_event", release), ("key_press_event", key)):
        fig.canvas.mpl_connect(nome, f)
    atualizar()
    _esperar(fig, est)
    plt.close(fig)
    if not est["ok"]:
        raise RuntimeError("Janela fechada antes de confirmar as retas (desenhe as duas e aperte Enter).")
    r = lambda v: round(float(v), 1)
    return dict(x_medial_mm=r(est["x_medial"]),
                linha_mm=((r(est["p1"][0]), r(est["p1"][1])), (r(est["p2"][0]), r(est["p2"][1]))))


def _desenhar_limites_sagitais(vol, corpo, spacing, k_ref, lado, titulo, L):
    """Vista sagital pela mama: o usuario desenha duas retas horizontais (limite superior e inferior)."""
    dx, dy, dz = spacing
    nz, ny, nx = vol.shape
    ys, _ = np.where(corpo[k_ref])
    (x1, y1), (x2, y2) = L["linha_mm"]
    X = (np.arange(nx) + 0.5) * dx
    lado_col = X < L["x_medial_mm"] if lado == "mama_direita" else X >= L["x_medial_mm"]
    y_lin = y1 + (X - x1) * (y2 - y1) / (x2 - x1)
    frente = np.where(corpo[k_ref].any(axis=0), y_lin - (np.argmax(corpo[k_ref], axis=0) + 0.5) * dy, -np.inf)
    col = int(np.argmax(np.where(lado_col, frente, -np.inf)))   # coluna em que a mama mais passa da linha

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(vol[:, :, col], cmap="bone", vmin=-200, vmax=300, origin="lower",
              extent=[0, ny * dy, 0, nz * dz], aspect="equal")
    ax.axvline(y_lin[col], color="orange", lw=1.5, ls="--", label="reta de corte nesta coluna")
    ax.axhline((k_ref + 0.5) * dz, color="yellow", ls=":", lw=1, label=f"fatia {k_ref}")
    ax.set_xlim(max(ys.min() * dy - 20, 0), min(y_lin[col] + 80, ny * dy))
    ax.set_xlabel("y (mm) -> posterior"); ax.set_ylabel("z (mm) -> superior")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_autoscale_on(False)
    h = [ax.axhline(np.nan, color="cyan", lw=2), ax.axhline(np.nan, color="cyan", lw=2)]
    faixa = [None]
    est = dict(z=[], arrastando=False, ok=False)

    def atualizar():
        n = len(est["z"])
        txt = ["Desenhe a reta do limite SUPERIOR da mama (clique e arraste para cima/baixo)",
               "Desenhe a reta do limite INFERIOR da mama",
               "Vermelho = faixa do recorte.  ENTER confirma | R refaz"][min(n, 2) if not est["arrastando"] else n - 1]
        ax.set_title(f"{titulo} - vista sagital (x = {X[col]:.0f} mm)\n{txt}", fontsize=10,
                     color="darkgreen" if n == 2 and not est["arrastando"] else "darkred")
        for i in range(2):
            h[i].set_ydata([est["z"][i]] * 2 if i < n else [np.nan] * 2)
        if faixa[0] is not None:
            faixa[0].remove(); faixa[0] = None
        if n == 2 and not est["arrastando"]:
            za, zb = sorted(est["z"])
            faixa[0] = ax.axhspan(za, zb, color="red", alpha=0.15)
        fig.canvas.draw_idle()

    def press(ev):
        if ev.inaxes is ax and ev.button == 1 and ev.ydata is not None and len(est["z"]) < 2:
            est["z"].append(ev.ydata); est["arrastando"] = True; atualizar()

    def move(ev):
        if est["arrastando"] and ev.inaxes is ax and ev.ydata is not None:
            est["z"][-1] = ev.ydata; atualizar()

    def release(ev):
        if est["arrastando"]:
            est["arrastando"] = False; atualizar()

    def key(ev):
        if ev.key in ("r", "R"):
            est.update(z=[], arrastando=False); atualizar()
        elif ev.key == "enter" and len(est["z"]) == 2:
            est["ok"] = True

    for nome, f in (("button_press_event", press), ("motion_notify_event", move),
                    ("button_release_event", release), ("key_press_event", key)):
        fig.canvas.mpl_connect(nome, f)
    atualizar()
    _esperar(fig, est)
    plt.close(fig)
    if not est["ok"]:
        raise RuntimeError("Janela fechada antes de confirmar os limites (desenhe as duas retas e aperte Enter).")
    za, zb = sorted(est["z"])
    return (round(float(za), 1), round(float(zb), 1))


def escolher_linhas_desenho(vol, corpo, spacing, k_ref):
    """
    Para cada mama o usuario DESENHA retas (clica, arrasta e solta):
      no corte axial  : reta 1 vertical (limite medial) e reta 2 de corte (inclinada);
      na vista sagital: duas retas horizontais (limites superior e inferior da mama).
    Em cada janela: R refaz, Enter confirma.
    Devolve {lado: dict(x_medial_mm=..., linha_mm=((x1, y1), (x2, y2)), z_mm=(z_inf, z_sup))}.
    """
    _verificar_backend_interativo()
    nomes = {"mama_direita": "MAMA DIREITA (lado esquerdo da tela)",
             "mama_esquerda": "MAMA ESQUERDA (lado direito da tela)"}
    linhas = {}
    for lado in LADOS:
        L = _desenhar_retas_axiais(vol, corpo, spacing, k_ref, lado, nomes[lado], linhas)
        L["z_mm"] = _desenhar_limites_sagitais(vol, corpo, spacing, k_ref, lado, nomes[lado], L)
        linhas[lado] = L
    return linhas


def separar_mamas_linhas(vol, corpo, spacing, linhas, protrusao_min_mm=10.0):
    """
    Separa as mamas com as linhas do usuario (as mesmas em todas as fatias):
      mama = corpo  E  lado certo da linha vertical  E  a frente (acima) da linha de corte.
    Entram so as fatias em que a mama passa pelo menos protrusao_min_mm a frente da linha.
    Devolve o mesmo dicionario de separar_mamas.
    """
    dx, dy, _ = spacing
    nz, ny, nx = corpo.shape
    X = (np.arange(nx) + 0.5) * dx                            # mm, centro do pixel
    Y = (np.arange(ny) + 0.5) * dy
    mascaras, fatias, meio_lado = {}, {}, {}
    for lado in LADOS:
        L = linhas[lado]
        (x1, y1), (x2, y2) = L["linha_mm"]
        y_linha = y1 + (X - x1) * (y2 - y1) / (x2 - x1)       # y da linha em cada coluna (mm)
        dist = y_linha[None, :] - Y[:, None]                  # > 0: a frente da linha (mm, vertical)
        lado_col = X < L["x_medial_mm"] if lado == "mama_direita" else X >= L["x_medial_mm"]
        regiao = (dist > 0) & lado_col[None, :]               # (Y, X), igual em todas as fatias
        m = corpo & regiao[None]
        if L.get("z_mm") is not None:                          # limites superior/inferior
            zc = (np.arange(nz) + 0.5) * spacing[2]
            m[(zc < L["z_mm"][0]) | (zc > L["z_mm"][1])] = False
        prot = np.where(m, dist[None], 0.0).max(axis=(1, 2))  # quanto a mama passa da linha, por fatia
        m[prot < protrusao_min_mm] = False
        rot, n = ndimage.label(m)
        if n > 1:
            tam = ndimage.sum_labels(np.ones_like(rot), rot, index=np.arange(1, n + 1))
            m = rot == (np.argmax(tam) + 1)
        mascaras[lado] = m
        fatias[lado] = np.where(m.any(axis=(1, 2)))[0]
        meio_lado[lado] = int(round(L["x_medial_mm"] / dx - 0.5)) + (0 if lado == "mama_direita" else 1)
        if fatias[lado].size == 0:
            raise RuntimeError(f"Nenhuma fatia da {lado.replace('_', ' ')} passa {protrusao_min_mm} mm da linha de "
                               "corte. Confira as linhas ou reduza PROTRUSAO_MIN_MM.")
    meio = int(round(np.mean(list(meio_lado.values()))))
    return dict(meio=meio, meio_lado=meio_lado, corte=None, linhas=linhas, mascaras=mascaras, fatias=fatias,
                perfil=np.where(np.isfinite(perfil_anterior(corpo)), perfil_anterior(corpo), ny).min(axis=0))


def caixa_da_mama(mascara, spacing, folga_mm=3.0, margem_mm=2.0, meio=None, lado=None):
    """Indices [k0,k1) x [r0,r1) x [c0,c1) do recorte; o lado interno termina na linha media."""
    dx, dy, _ = spacing
    ks = np.where(mascara.any(axis=(1, 2)))[0]
    rs = np.where(mascara.any(axis=(0, 2)))[0]
    cs = np.where(mascara.any(axis=(0, 1)))[0]
    nz, ny, nx = mascara.shape
    r0 = max(rs[0] - int(round(folga_mm / dy)), 0)            # camada de agua acima do mamilo
    r1 = min(rs[-1] + 1, ny)
    mg = int(round(margem_mm / dx))
    if lado == "mama_direita":
        c0, c1 = max(cs[0] - mg, 0), meio
    else:
        c0, c1 = meio, min(cs[-1] + 1 + mg, nx)
    return ks[0], ks[-1] + 1, r0, r1, c0, c1


def figura_separacao(vol, sep, spacing, k_meio, arquivo, mostrar):
    dx, dy, _ = spacing
    ny, nx = vol.shape[1:]
    ext = [0, nx * dx, ny * dy, 0]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    ax[0].imshow(vol[k_meio], cmap="bone", vmin=-200, vmax=300, extent=ext)
    cores = {"mama_direita": "tab:red", "mama_esquerda": "tab:cyan"}
    for lado in LADOS:
        m = sep["mascaras"][lado][k_meio]
        if m.any():
            ax[0].contour(m, levels=[0.5], colors=cores[lado], linewidths=1.5, extent=[0, nx * dx, 0, ny * dy])
    if sep.get("linhas"):
        for i, (lado, L) in enumerate(sep["linhas"].items()):
            (x1, y1), (x2, y2) = L["linha_mm"]
            xa = np.array([0.0, L["x_medial_mm"]]) if lado == "mama_direita" else np.array([L["x_medial_mm"], nx * dx])
            ya = y1 + (xa - x1) * (y2 - y1) / (x2 - x1)
            ax[0].axvline(L["x_medial_mm"], color="yellow", ls="--", lw=1, label="limite medial" if i == 0 else None)
            ax[0].plot(xa, ya, color="orange", lw=1.5, label="linha de corte" if i == 0 else None)
            ax[0].plot([x1, x2], [y1, y2], "o", color="orange", ms=4)
    else:
        ax[0].axvline((sep["meio"] + 0.5) * dx, color="yellow", ls="--", lw=1, label="linha media")
        if np.isfinite(sep["corte"][k_meio]):
            ax[0].axhline((sep["corte"][k_meio] + 0.5) * dy, color="orange", ls=":", lw=1, label="limite posterior")
    ax[0].set_title(f"Fatia {k_meio}: vermelho = mama direita, ciano = mama esquerda", fontsize=11)
    ax[0].set_xlabel("x (mm) -> esquerda do paciente"); ax[0].set_ylabel("y (mm) -> posterior")
    ax[0].legend(loc="lower right", fontsize=9)
    fundo = max(np.where(sep["mascaras"][l].any(axis=(0, 2)))[0].max() for l in LADOS)
    ax[0].set_ylim(min((fundo + 25) * dy, ny * dy), max((sep["perfil"].min() - 15) * dy, 0))

    # projecao frontal (coronal) da extensao das mamas
    proj = np.zeros(vol.shape[::2], dtype=np.int8)            # (Z, X)
    for i, lado in enumerate(LADOS, start=1):
        proj[sep["mascaras"][lado].any(axis=1)] = i
    ax[1].imshow(proj, cmap="viridis", origin="lower", aspect="equal",
                 extent=[0, nx * dx, 0, vol.shape[0] * spacing[2]])
    for l in LADOS:
        ax[1].axvline((sep["meio_lado"][l] + 0.5) * dx, color="yellow", ls="--", lw=1)
        L = (sep.get("linhas") or {}).get(l)
        if L and L.get("z_mm") is not None:
            xa = (0, L["x_medial_mm"]) if l == "mama_direita" else (L["x_medial_mm"], nx * dx)
            for z in L["z_mm"]:
                ax[1].plot(xa, (z, z), color="orange", lw=1.5)
    ax[1].set_title("Projecao coronal: 1 = direita, 2 = esquerda", fontsize=11)
    ax[1].set_xlabel("x (mm)"); ax[1].set_ylabel("z (mm) -> superior")
    plt.tight_layout()
    plt.savefig(arquivo, dpi=90)
    if mostrar:
        plt.show()
    plt.close(fig)


# ====================================================================== #
# 3b. Inclusao (lesao) a partir do PET do mesmo estudo
# ====================================================================== #
def carregar_suv(pasta_pet, img_ct):
    """
    Le o PET (FDG), converte para SUV de peso corporal e reamostra na grade da TC.
    PET e TC do mesmo estudo PET/CT compartilham o Frame of Reference: basta reamostrar.
    """
    import pydicom
    reader = sitk.ImageSeriesReader()
    nomes = reader.GetGDCMSeriesFileNames(pasta_pet)
    if not nomes:
        raise FileNotFoundError(f"Nenhum DICOM de PET em: {pasta_pet}")
    reader.SetFileNames(nomes)
    pet = reader.Execute()
    ds = pydicom.dcmread(nomes[0], stop_before_pixels=True)
    bq = sitk.GetArrayFromImage(sitk.Resample(pet, img_ct, sitk.Transform(), sitk.sitkLinear, 0.0)).astype(np.float32)
    try:
        rp = ds.RadiopharmaceuticalInformationSequence[0]
        peso_g = float(ds.PatientWeight) * 1000.0
        dose = float(rp.RadionuclideTotalDose)
        meia_vida = float(rp.RadionuclideHalfLife)
        t_inj = datetime.datetime.strptime(str(rp.RadiopharmaceuticalStartTime).split(".")[0], "%H%M%S")
        t_ser = datetime.datetime.strptime(str(ds.SeriesTime).split(".")[0], "%H%M%S")
        # DecayCorrection = START: atividade corrigida para o inicio da serie
        dose_corr = dose * 2.0 ** (-(t_ser - t_inj).total_seconds() / meia_vida)
        if str(ds.Units).upper() != "BQML":
            raise ValueError(f"unidade {ds.Units}")
        return bq * peso_g / dose_corr, "SUV"
    except Exception as e:                                      # sem dados para SUV: usa o valor bruto
        print(f"   (aviso: SUV nao calculado ({e}); usando a intensidade bruta do PET)")
        return bq, "PET bruto"


def segmentar_inclusoes(suv, mascaras, corpo, spacing, fracao_max=0.40, suv_minimo=2.5,
                        raio_max_mm=40.0):
    """
    Para cada mama procura o pico de captacao DENTRO da mama (sem a pele). Buscar alem da mama
    faria o pico cair no coracao, que capta muito FDG. Se o pico for >= suv_minimo, a inclusao
    e o componente conexo com SUV > fracao_max * SUVmax que contem o pico (limiar classico de
    40%), limitado a raio_max_mm do pico (a lesao pode passar da base da mama, mas nao pode
    "escorrer" para outros orgaos). Devolve {lado: mascara bool ou None} e um resumo.
    """
    dx, dy, dz = spacing
    incl, resumo = {}, {}
    zz, yy, xx = np.indices(suv.shape, sparse=True)
    for lado in LADOS:
        busca = mascaras[lado] & ndimage.binary_erosion(corpo, np.ones((1, 3, 3), bool), iterations=2)
        if not busca.any():
            incl[lado] = None
            continue
        v = np.where(busca, suv, -np.inf)
        k, y, x = np.unravel_index(np.argmax(v), v.shape)
        suv_max = float(suv[k, y, x])
        if suv_max < suv_minimo:
            incl[lado] = None
            resumo[lado] = dict(suv_max=suv_max)
            continue
        perto = ((zz - k) * dz) ** 2 + ((yy - y) * dy) ** 2 + ((xx - x) * dx) ** 2 <= raio_max_mm ** 2
        rot, _ = ndimage.label((suv > fracao_max * suv_max) & corpo & perto)
        m = rot == rot[k, y, x]
        incl[lado] = m
        fz = np.where(m.any(axis=(1, 2)))[0]
        resumo[lado] = dict(suv_max=suv_max, pico=(int(k), int(y), int(x)), volume_ml=m.sum() * dx * dy * dz / 1000,
                            fatias=(int(fz[0]), int(fz[-1])), dentro=float((m & mascaras[lado]).sum() / m.sum()))
    return incl, resumo


# ====================================================================== #
# 5. Simulacao de uma mama
# ====================================================================== #
def simular_mama(vol_c, corpo_c, mascara_c, spacing, sim, mascarar_saida=True, rotulo="",
                 inclusao_c=None, ecogenicidade_inclusao=0.1):
    """
    Simula o volume recortado de uma mama. Devolve (vol_db [Z, linhas, colunas] em [-DR, 0],
    prof (mm, a partir do topo do recorte), xs (mm, a partir da borda esquerda do recorte)).
    """
    nz, ny, nx = vol_c.shape
    dx, dy, _ = spacing
    t0 = time.time()
    rn, m, _ = sim.preprocessar_volume(vol_c, corpo_c, spacing, roi_3d=mascara_c.astype(np.float32))
    if inclusao_c is not None and inclusao_c.any():
        # Inclusao (tumor): hipoecoica. A TC nao distingue bem tumor de glandula (HU parecidos);
        # o PET define a regiao e aqui ela recebe ecogenicidade propria no mapa de espalhamento.
        # A borda continua com reflexao, porque a impedancia vem do HU da TC.
        m = np.where(inclusao_c, np.float32(ecogenicidade_inclusao), m)
    coords, prof, xs = sim.coordenadas_regiao(spacing, 0.0, nx * dx, 0.0, ny * dy)
    uns = np.ones((ny, nx), np.float32)
    saida = None
    for k in range(nz):
        Ius, _, _ = sim.simular_fatia(vol_c[k], corpo_c[k], uns, rn[k], m[k], coords, spacing)
        Ius = Ius[::2]
        if mascarar_saida:
            dentro = ndimage.map_coordinates(mascara_c[k].astype(np.float32), coords[:, ::2], order=1) > 0.5
            Ius = np.where(dentro, Ius, 0.0)
        if saida is None:
            saida = np.zeros((nz,) + Ius.shape, np.float32)
        saida[k] = Ius
        print(f"\r   -> {rotulo}: fatia {k + 1}/{nz}", end="", flush=True)
    print(f"  ({time.time() - t0:.0f} s)")
    vol_db = sim.comprimir_log(saida)
    return vol_db, prof[::2], xs


# ====================================================================== #
# 6. Gravacao DICOM
# ====================================================================== #
_RAIZ_UID = "1.2.826.0.1.3680043.2.1125."     # raiz usada nos exemplos do SimpleITK/GDCM


def novo_uid():
    agora = datetime.datetime.now()
    return _RAIZ_UID + agora.strftime("%Y%m%d%H%M%S") + "." + str(agora.microsecond) + str(np.random.randint(1000, 9999))


def escrever_serie_dicom(vol, origem, espacamento, tags_base, pasta, descricao, modalidade,
                         numero_serie, extras=None):
    """
    Grava vol [Z, Y, X] (uint8 ou int16) como uma serie DICOM, uma fatia por arquivo, com
    orientacao LPS (identidade), origem = centro do voxel [0, 0, 0] em mm (x, y, z).
    """
    os.makedirs(pasta, exist_ok=True)
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing([float(s) for s in espacamento])
    img.SetOrigin([float(o) for o in origem])
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    agora = datetime.datetime.now()
    data, hora = agora.strftime("%Y%m%d"), agora.strftime("%H%M%S")
    copiar = ["0010|0010", "0010|0020", "0010|0030", "0010|0040", "0010|1010",  # paciente
              "0020|000d", "0008|0020", "0008|0030", "0008|0050", "0008|1030",  # estudo
              "0020|0010", "0020|0052", "0018|5100", "0008|0070"]
    serie = [(t, tags_base[t]) for t in copiar if t in tags_base]
    serie += [
        ("0020|000e", novo_uid()),                       # Series Instance UID
        ("0008|103e", descricao),                        # Series Description
        ("0008|0060", modalidade),
        ("0020|0011", str(numero_serie)),
        ("0008|0008", "DERIVED\\SECONDARY"),
        ("0008|0021", data), ("0008|0031", hora),
        ("0008|0064", "WSD"),                            # Conversion Type: workstation
        ("0020|0037", "1\\0\\0\\0\\1\\0"),               # Image Orientation (Patient): LPS
        ("0018|0050", f"{espacamento[2]:.4f}"),
        # Pixel Spacing explicito (linha\\coluna): sem isso a classe US grava 1 x 1 mm
        ("0028|0030", f"{espacamento[1]:.6f}\\{espacamento[0]:.6f}"),
        ("0008|2111", "Simulacao TC->US (Kutter, Shams & Navab 2009) - CT_para_US_mamas.py"),
    ]
    serie += list((extras or {}).items())

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for i in range(img.GetDepth()):
        fatia = img[:, :, i]
        for t, v in serie:
            fatia.SetMetaData(t, v)
        fatia.SetMetaData("0008|0012", data)
        fatia.SetMetaData("0008|0013", hora)
        fatia.SetMetaData("0008|0018", novo_uid() + f".{i + 1}")       # SOP Instance UID
        fatia.SetMetaData("0020|0032", "\\".join(f"{c:.4f}" for c in img.TransformIndexToPhysicalPoint((0, 0, i))))
        fatia.SetMetaData("0020|1041", f"{img.TransformIndexToPhysicalPoint((0, 0, i))[2]:.4f}")
        fatia.SetMetaData("0020|0013", str(i + 1))
        writer.SetFileName(os.path.join(pasta, f"{i + 1:04d}.dcm"))
        writer.Execute(fatia)


# ====================================================================== #
# Programa principal
# ====================================================================== #
def processar(caminho_serie, pasta_saida, offset_corte_mm=10.0, protrusao_min_mm=10.0,
              meio_coluna=None, salvar_ct=True, mascarar_ct=True, confirmar=True,
              mostrar_figuras=True, parametros_simulador=None, pasta_pet=None,
              ecogenicidade_inclusao=0.1, fracao_suv=0.40, suv_minimo=2.5,
              modo_corte="automatico", linhas_corte=None, fatia_referencia_corte=None):
    os.makedirs(pasta_saida, exist_ok=True)
    print("1. Lendo a serie DICOM e reorientando para LPS...")
    img, vol, tags = carregar_serie(caminho_serie)
    spacing = img.GetSpacing()                                    # (dx, dy, dz)
    print(f"   -> {vol.shape} (fatias x linhas x colunas), voxel {tuple(round(s, 3) for s in spacing)} mm, "
          f"posicao do paciente: {tags.get('0018|5100', '?').strip()}")

    print("2. Segmentando o corpo...")
    corpo = SimuladorKutter.mascara_corpo(vol)

    print("3. Separando mama direita e esquerda...")
    if modo_corte in ("desenhar", "clique", "manual"):
        if modo_corte in ("desenhar", "clique"):
            k_ref_corte = fatia_referencia_corte if fatia_referencia_corte is not None else fatia_referencia(corpo)
            linhas_corte = escolher_linhas_desenho(vol, corpo, spacing, k_ref_corte)
            print(f"   -> linhas escolhidas na fatia {k_ref_corte}. Para repetir sem clicar, use no __main__:")
            print(f"      MODO_CORTE = \"manual\"\n      LINHAS_CORTE = {linhas_corte}")
        elif not linhas_corte:
            raise ValueError("MODO_CORTE = 'manual' precisa de LINHAS_CORTE.")
        sep = separar_mamas_linhas(vol, corpo, spacing, linhas_corte, protrusao_min_mm)
    else:
        sep = separar_mamas(vol, corpo, spacing, offset_corte_mm, protrusao_min_mm, meio_coluna)
    for lado in LADOS:
        f = sep["fatias"][lado]
        print(f"   -> {lado.replace('_', ' ')}: fatias {f[0]} a {f[-1]} ({f.size}), "
              f"volume {sep['mascaras'][lado].sum() * np.prod(spacing) / 1000:.0f} cm3")
    if sep.get("linhas"):
        for lado in LADOS:
            L = sep["linhas"][lado]
            (x1, y1), (x2, y2) = L["linha_mm"]
            ang = np.degrees(np.arctan(abs(y2 - y1) / abs(x2 - x1)))
            medial_mais_alto = (y1 if abs(x1 - L["x_medial_mm"]) < abs(x2 - L["x_medial_mm"]) else y2) < max(y1, y2)
            print(f"   -> {lado.replace('_', ' ')}: limite medial x = {L['x_medial_mm']:.1f} mm, linha de corte de "
                  f"({x1:.0f}, {y1:.0f}) a ({x2:.0f}, {y2:.0f}) mm ({ang:.0f} graus, "
                  f"{'descendo para a lateral' if medial_mais_alto else 'subindo para a lateral'}), "
                  f"fatias {sep['fatias'][lado][0]}-{sep['fatias'][lado][-1]}")
    else:
        print(f"   -> linha media na coluna {sep['meio']} (x = {(sep['meio'] + 0.5) * spacing[0]:.1f} mm do canto)")
    inclusoes = {lado: None for lado in LADOS}
    if pasta_pet:
        print("3b. Procurando inclusoes no PET (FDG)...")
        suv, unid = carregar_suv(pasta_pet, img)
        inclusoes, resumo = segmentar_inclusoes(suv, sep["mascaras"], corpo, spacing, fracao_suv, suv_minimo)
        for lado in LADOS:
            r_ = resumo.get(lado, {})
            if inclusoes[lado] is None:
                print(f"   -> {lado.replace('_', ' ')}: sem inclusao ({unid} max {r_.get('suv_max', float('nan')):.2f} < {suv_minimo})")
                continue
            print(f"   -> {lado.replace('_', ' ')}: INCLUSAO {unid} max {r_['suv_max']:.1f} (fatia {r_['pico'][0]}), "
                  f"{r_['volume_ml']:.1f} mL, fatias {r_['fatias'][0]}-{r_['fatias'][1]}, "
                  f"{100 * r_['dentro']:.0f}% dentro do recorte original")
            # a mama passa a incluir a lesao (+2 mm), mesmo que ela fique alem do limite posterior
            margem = ndimage.binary_dilation(inclusoes[lado], np.ones((1, 3, 3), bool), iterations=2) & corpo
            lado_ok = (np.arange(vol.shape[2]) < sep["meio_lado"][lado]) if lado == "mama_direita" \
                else (np.arange(vol.shape[2]) >= sep["meio_lado"][lado])
            sep["mascaras"][lado] = sep["mascaras"][lado] | (margem & lado_ok[None, None, :])
            inclusoes[lado] &= lado_ok[None, None, :]
    k_ref = int(np.argmax([sep["mascaras"]["mama_direita"][k].sum() + sep["mascaras"]["mama_esquerda"][k].sum()
                           for k in range(vol.shape[0])]))
    arq_fig = os.path.join(pasta_saida, "separacao_mamas.png")
    figura_separacao(vol, sep, spacing, k_ref, arq_fig, mostrar_figuras)
    print(f"   -> figura de conferencia: {arq_fig}")
    if confirmar:
        resp = input("   A separacao esta correta? [s/n]: ").strip().lower()
        if resp not in ("s", "sim", "y", ""):
            print("   Ajuste MODO_CORTE / LINHAS_CORTE (ou OFFSET_CORTE_MM, PROTRUSAO_MIN_MM, MEIO_COLUNA no modo\n"
                  "   automatico) no inicio do __main__ e rode de novo.")
            return None

    resultados = {}
    for n_lado, lado in enumerate(LADOS):
        print(f"4-6. {lado.replace('_', ' ').upper()}")
        k0, k1, r0, r1, c0, c1 = caixa_da_mama(sep["mascaras"][lado], spacing, meio=sep["meio_lado"][lado], lado=lado)
        vol_c = vol[k0:k1, r0:r1, c0:c1]
        corpo_c = corpo[k0:k1, r0:r1, c0:c1]
        masc_c = sep["mascaras"][lado][k0:k1, r0:r1, c0:c1]
        incl_c = None if inclusoes[lado] is None else inclusoes[lado][k0:k1, r0:r1, c0:c1]
        img_c = img[c0:c1, r0:r1, k0:k1]                          # mantem a geometria fisica
        print(f"   -> recorte: {vol_c.shape} voxels "
              f"({(c1 - c0) * spacing[0]:.0f} x {(r1 - r0) * spacing[1]:.0f} x {(k1 - k0) * spacing[2]:.0f} mm)")

        sim = SimuladorKutter(**(parametros_simulador or {}))
        vol_db, prof, xs = simular_mama(vol_c, corpo_c, masc_c, spacing, sim, rotulo=lado.replace("_", " "),
                                        inclusao_c=incl_c, ecogenicidade_inclusao=ecogenicidade_inclusao)

        # geometria: centro do pixel (linha j, coluna i) do US em coordenadas do paciente
        o = img_c.GetOrigin()                                      # centro do voxel [0,0,0] do recorte
        origem_us = (o[0] + xs[0] - 0.5 * spacing[0], o[1] + prof[0] - 0.5 * spacing[1], o[2])
        esp_us = (sim.dx_fino, sim.dz_fino * 2, spacing[2])
        dr = sim.faixa_dinamica_db
        us8 = np.round((vol_db + dr) / dr * 255.0).astype(np.uint8)
        pasta_lado = os.path.join(pasta_saida, lado)
        nome = "direita" if lado == "mama_direita" else "esquerda"
        escrever_serie_dicom(us8, origem_us, esp_us, tags, os.path.join(pasta_lado, "US"),
                             f"US simulado (Kutter 2009) - mama {nome}", "US", 900 + 2 * n_lado,
                             extras={"0028|1050": "128", "0028|1051": "256",
                                     "0018|6030": str(sim.f0_mhz * 1e6)})
        print(f"   -> DICOM US: {os.path.join(pasta_lado, 'US')} ({us8.shape[0]} fatias, "
              f"{us8.shape[2]} x {us8.shape[1]} pixels de {esp_us[0]:.3f} x {esp_us[1]:.3f} mm)")

        if salvar_ct:
            ct = vol_c.copy()
            if mascarar_ct:
                ct[~masc_c] = -1000.0
            escrever_serie_dicom(np.round(ct).astype(np.int16), img_c.GetOrigin(), spacing, tags,
                                 os.path.join(pasta_lado, "CT"), f"TC recortada - mama {nome}", "CT",
                                 901 + 2 * n_lado,
                                 extras={"0028|1052": "0", "0028|1053": "1", "0028|1050": "40", "0028|1051": "400"})
            print(f"   -> DICOM TC: {os.path.join(pasta_lado, 'CT')}")
        if incl_c is not None:
            # mapa de rotulos da inclusao (0/1) na grade da TC recortada, para conferir no Slicer
            escrever_serie_dicom(incl_c.astype(np.uint8), img_c.GetOrigin(), spacing, tags,
                                 os.path.join(pasta_lado, "INCLUSAO"), f"Inclusao (PET {int(100 * fracao_suv)}% SUVmax) - mama {nome}",
                                 "OT", 905 + n_lado, extras={"0028|1050": "0.5", "0028|1051": "1"})
            print(f"   -> DICOM inclusao: {os.path.join(pasta_lado, 'INCLUSAO')}")
        resultados[lado] = dict(us_db=vol_db, ct=vol_c, mascara=masc_c, prof=prof, xs=xs, inclusao=incl_c,
                                caixa=(k0, k1, r0, r1, c0, c1), dr=dr)

    # figura final: TC recortada x US simulado na fatia de maior area de cada mama
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    for j, lado in enumerate(LADOS):
        R = resultados[lado]
        alvo = R["inclusao"] if R["inclusao"] is not None else R["mascara"]
        k = int(np.argmax(alvo.sum(axis=(1, 2))))
        ny_c, nx_c = R["ct"].shape[1:]
        ax[0, j].imshow(np.where(R["mascara"][k], R["ct"][k], -1000), cmap="bone", vmin=-200, vmax=200,
                        extent=[0, nx_c * spacing[0], ny_c * spacing[1], 0])
        ax[0, j].set_title(f"TC recortada - {lado.replace('_', ' ')} (fatia {R['caixa'][0] + k})")
        if R["inclusao"] is not None and R["inclusao"][k].any():
            for a in (ax[0, j], ax[1, j]):
                a.contour(R["inclusao"][k], [0.5], colors="lime", linewidths=1.2,
                          extent=[0, nx_c * spacing[0], 0, ny_c * spacing[1]])
        ax[1, j].imshow(R["us_db"][k], cmap="gray", vmin=-R["dr"], vmax=0,
                        extent=[R["xs"][0], R["xs"][-1], R["prof"][-1], R["prof"][0]])
        ax[1, j].set_title(f"US simulado - {lado.replace('_', ' ')}")
        for a in (ax[0, j], ax[1, j]):
            a.set_xlabel("x (mm)"); a.set_ylabel("profundidade (mm)")
    plt.tight_layout()
    arq = os.path.join(pasta_saida, "resultado_mamas.png")
    plt.savefig(arq, dpi=80)
    print(f"Figura final: {arq}")
    if mostrar_figuras:
        plt.show()
    plt.close(fig)
    return resultados


if __name__ == "__main__":
    PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))

    # ------------------------- CONFIGURACAO -------------------------
    CAMINHO_SERIE = os.path.join(PASTA_SCRIPT, "2-CTAC-14691")   # serie CTAC do QIN-BREAST
    PASTA_SAIDA = os.path.join(PASTA_SCRIPT, "saida_US_mamas")
    # CORTE DAS MAMAS
    #   "desenhar"  : em cada mama o usuario DESENHA as retas (clica, arrasta e solta):
    #                 no corte axial, a reta vertical (limite medial) e a reta de corte inclinada
    #                 (acompanhando a parede toracica); na vista sagital, as retas dos limites superior
    #                 e inferior. R refaz, Enter confirma. O programa imprime LINHAS_CORTE para repetir.
    #   "manual"    : usa LINHAS_CORTE (em mm, mesmos eixos das figuras: x para a direita, y para baixo)
    #   "automatico": sulco intermamario + linha horizontal OFFSET_CORTE_MM alem da pele (versao anterior)
    MODO_CORTE = "desenhar"
    LINHAS_CORTE = None         # ex. (serie QIN-BREAST-01-0002):
    # LINHAS_CORTE = {"mama_direita":  {"x_medial_mm": 364.0, "linha_mm": ((362.0, 365.0), (215.0, 405.0)),
    #                                   "z_mm": (85.0, 185.0)},
    #                 "mama_esquerda": {"x_medial_mm": 367.0, "linha_mm": ((369.0, 365.0), (505.0, 400.0)),
    #                                   "z_mm": (100.0, 195.0)}}
    #   x_medial_mm: linha vertical (limite medial); linha_mm: dois pontos da linha de corte inclinada;
    #   z_mm: limites inferior e superior da mama (opcional; sem ele valem todas as fatias que passam da linha)
    FATIA_REFERENCIA_CORTE = None   # fatia mostrada para o clique (None = fatia do mamilo mais anterior)
    OFFSET_CORTE_MM = 10.0      # (automatico) quanto alem da pele da linha media entra no recorte
    PROTRUSAO_MIN_MM = 10.0     # fatias em que a mama passa menos que isso da linha de corte ficam de fora
    MEIO_COLUNA = None          # (automatico) coluna da linha media (None = pelo sulco intermamario)
    SALVAR_CT = True            # grava tambem a TC recortada de cada mama
    MASCARAR_CT = True          # na TC gravada, fora da mama = -1000 HU
    CONFIRMAR = True            # pergunta se a separacao esta correta antes de simular
    MOSTRAR_FIGURAS = True
    # Parametros do simulador (qualquer argumento de SimuladorKutter pode entrar aqui)
    PARAMETROS_SIMULADOR = dict(
        f0_mhz=5.0,
        faixa_dinamica_db=35.0,            # exibicao: 30-40 dB separa gordura, glandula e lesao
        ecogenicidade_minima=0.05,         # gordura: quanto menor, mais escura (glandula ~ +12 dB)
        alpha_mistura=0.0009,              # peso do speckle em relacao as reflexoes [Eq. 6]
        elementos_ativos_espalhamento=12,  # janela de Hann no speckle (0 = sem, grao mais fino)
        normalizar_pico_reflexao=True,     # reflexoes com pico preservado (correcao)
        suavizacao_ct_z_mm=1.5,            # TC suavizada para calcular Z (sem reflexoes do ruido)
    )
    # Inclusao a partir do PET do mesmo estudo (None = sem inclusao)
    PASTA_PET = os.path.join(PASTA_SCRIPT, "3-PET AC 3DWB-52072")
    ECOGENICIDADE_INCLUSAO = 0.1   # mapa de espalhamento na lesao (gordura ~0.2, glandula ~1): hipoecoica
    FRACAO_SUV = 0.40              # limiar de segmentacao: 40% do SUVmax
    SUV_MINIMO = 2.5               # abaixo disso considera que a mama nao tem inclusao
    # ----------------------------------------------------------------

    processar(CAMINHO_SERIE, PASTA_SAIDA, OFFSET_CORTE_MM, PROTRUSAO_MIN_MM, MEIO_COLUNA,
              SALVAR_CT, MASCARAR_CT, CONFIRMAR, MOSTRAR_FIGURAS, PARAMETROS_SIMULADOR,
              PASTA_PET if os.path.isdir(PASTA_PET) else None, ECOGENICIDADE_INCLUSAO, FRACAO_SUV, SUV_MINIMO,
              modo_corte=MODO_CORTE, linhas_corte=LINHAS_CORTE, fatia_referencia_corte=FATIA_REFERENCIA_CORTE)
