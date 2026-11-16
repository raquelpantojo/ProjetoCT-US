"""
Referencia do banco de dados:
Li, X., Abramson, R. G., Arlinghaus, L. R., Chakravarthy, A. B., Abramson, V. G., Sanders, M., & Yankeelov, T. E. (2016).
Data From QIN-BREAST (Version 2) [Data set].
The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.21JUEBH0

Referencias do modelo de simulacao:
[B&D] Bamber, J. C., & Dickinson, R. J. (1980). Ultrasonic B-scanning: a computer simulation.
      Phys. Med. Biol., 25(3), 463-479.
[DLD] Dillenseger, J.-L., Laguitton, S., & Delabrousse, E. (2009). Fast simulation of ultrasound
      images from a CT volume. Comput. Biol. Med., 39, 180-186.

Instalar:
pip install SimpleITK scipy numpy matplotlib pyvista

Versao "referencias" (derivada de TransformCTUS_revisado.py)
-------------------------------------------------------------
O que mudou em relacao a versao revisada (as opcoes antigas continuam disponiveis):

A) Funcao de ecogenicidade  (parametro modelo_ecogenicidade)
   - "derivada2"  [DLD Eq. 4-5]:  T = (1 / 2 Z0) * d2Z/dz2  (derivada 2a ao longo do feixe).
                  Como em [DLD, sec. 3.1], a derivada e aplicada na PSF e nao no mapa de Z
                  (I = (d2H/dz2) * Z), o que evita derivar numericamente os espalhadores pontuais.
   - "reflexao"   (versao anterior): R = (Z2 - Z1)/(Z2 + Z1) ~ (1/2) d(lnZ)/dz.
                  E o modelo 1D de impediografia (Jones 1975), que [B&D, Apendice 3] cita como distinto.

B) Modelo de espalhadores sub-resolucao  (parametro modelo_espalhadores)
   - "dillenseger" [DLD Eq. 7-8, sec. 2.2]: modelo de regularidade marcada (Cramblitt & Parker 1999):
                  distancias entre espalhadores ~ Gama(alpha, beta), amplitudes ~ log-normal
                  (media 1, variancia 0,1). A linha 1D e mapeada no plano por uma curva de Hilbert
                  (Laporte et al. 2007). Parametros por tecido: densidade e regularidade alpha.
                  Simplificacao: em vez de uma quadtree por regiao homogenea, usa-se uma unica curva
                  de Hilbert cobrindo a janela e mantem-se, para cada tecido, apenas os espalhadores
                  que caem nos seus pixels (a localidade da curva preserva a regularidade).
   - "bamber"     [B&D sec. 2.3, Eq. 5 e 7]: campo de compressibilidade aleatorio gaussiano com
                  autocorrelacao N(R) = exp(-R^2/a^2), sintetizado por filtragem de ruido branco
                  (equivalente a sintese de Fourier com fases aleatorias). Parametros: comprimento
                  de correlacao a e desvio RMS por tecido.
   - "branco"     (versao anterior): ruido gaussiano branco multiplicativo por amostra.

C) Mantidos da versao revisada (extensoes que NAO estao em [B&D] nem em [DLD]; [B&D] assume
   atenuacao nula e TGC perfeito e [DLD, sec. 3.2] diz que sombra e atenuacao estao ausentes):
   atenuacao ida-e-volta, perda de transmissao prod(1 - R^2), TGC, compressao log.
   Citar fontes proprias para essas etapas.

Notas de convencao da PSF:
   - Axial: sigma_z = c * sigma_t / 2 com banda fracionaria medida a -6 dB (amplitude), na
     coordenada de PROFUNDIDADE. A Eq. 9 de [DLD], sigma_ax = lambda*Q*sqrt(ln2)/pi, usa banda a
     -3 dB (potencia) e coordenada de percurso. Para comparar, use self.sigma_ax_dillenseger().
   - Lateral: FWHM ~ lambda * F# invariante com a profundidade, como em [B&D] (regiao pequena).
     [DLD Eq. 10] faz a largura crescer linearmente com a profundidade (nao implementado aqui).
"""

import os
import time
import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import hilbert, fftconvolve
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, LassoSelector
from matplotlib.path import Path
import SimpleITK as sitk

try:
    import pyvista as pv
except ImportError:  # permite rodar o pipeline sem a visualizacao 3D
    pv = None


# Propriedades do meio de acoplamento (agua, ~20-25 C)
RHO_AGUA, C_AGUA, ALPHA_AGUA = 1000.0, 1480.0, 0.0022
Z_AGUA = RHO_AGUA * C_AGUA          # tambem usada como impedancia de referencia Z0 [DLD Eq. 4]

# Classes de tecido (mesmos limiares em todo o codigo)
AGUA, AR, GORDURA, GLANDULAR, OSSO = 0, 1, 2, 3, 4
NOMES_CLASSES = {AGUA: "agua", AR: "ar", GORDURA: "gordura", GLANDULAR: "fibroglandular", OSSO: "osso"}

# Parametros dos espalhadores por tecido (ad hoc, como em [DLD sec. 3.1]; ajustar/calibrar).
#   sigma        : desvio RMS relativo da impedancia (nivel de ecogenicidade do speckle)
#   a_corr_mm    : comprimento de correlacao do modelo de Bamber [B&D Eq. 5]
#   densidade_mm2: espalhadores por mm^2 no modelo de Dillenseger
#   alpha        : parametro de forma da Gama (regularidade): <1 agrupado, =1 Poisson, >1 regular
# Em [DLD]: gordura -> alta densidade e regular (alpha=10); figado (tecido mole) -> media
# densidade e regularidade (alpha=0,4); agua/ar -> baixa densidade e irregular (alpha=0,1).
# Atencao ao a_corr_mm (Bamber): o campo e normalizado para RMS 1, mas so a parte do espectro na
# banda da PSF (~2/lambda ciclos/mm) gera eco. Com f0 = 5 MHz, o nivel do speckle cai ~3 dB de
# a = 0,04 para 0,06 mm, ~13 dB ate 0,12 mm e ~32 dB ate 0,20 mm (tecido quase anecoico).
PARAMS_ESPALHADORES = {
    AGUA:      dict(sigma=0.000, a_corr_mm=0.05, densidade_mm2=0.0,  alpha=0.1),
    AR:        dict(sigma=0.000, a_corr_mm=0.05, densidade_mm2=0.0,  alpha=0.1),
    GORDURA:   dict(sigma=0.010, a_corr_mm=0.07, densidade_mm2=60.0, alpha=10.0),
    GLANDULAR: dict(sigma=0.025, a_corr_mm=0.05, densidade_mm2=50.0, alpha=0.4),
    OSSO:      dict(sigma=0.010, a_corr_mm=0.05, densidade_mm2=30.0, alpha=1.0),
}


# ---------------------------------------------------------------------- #
# Curva de Hilbert (indice d de cada ponto (x, y) numa grade n x n, n = 2^k)
# ---------------------------------------------------------------------- #
def hilbert_xy2d(n: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Versao vetorizada do algoritmo classico xy2d (n potencia de 2)."""
    x = x.astype(np.int64).copy()
    y = y.astype(np.int64).copy()
    d = np.zeros_like(x)
    s = n // 2
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        # rotacao do quadrante
        flip = (ry == 0) & (rx == 1)
        x = np.where(flip, n - 1 - x, x)
        y = np.where(flip, n - 1 - y, y)
        troca = ry == 0
        x, y = np.where(troca, y, x), np.where(troca, x, y)
        s //= 2
    return d


class PipelineTCparaUltrassom:
    """
    Conversao de volumes DICOM de Tomografia em imagens B-mode de Ultrassom.
    """

    def __init__(self, f0_mhz: float = 5.0, c0_m_s: float = 1540.0,
                 banda_fracionaria: float = 0.6, f_number: float = 2.0,
                 largura_sonda_mm: float = 36.8, profundidade_mm: float = 50.0,
                 faixa_dinamica_db: float = 50.0, tgc_db_cm_mhz: float = 0.5,
                 modelo_ecogenicidade: str = "derivada2",
                 modelo_espalhadores: str = "dillenseger",
                 params_espalhadores: dict = None,
                 semente: int = 0):
        if modelo_ecogenicidade not in ("derivada2", "reflexao"):
            raise ValueError("modelo_ecogenicidade deve ser 'derivada2' ou 'reflexao'")
        if modelo_espalhadores not in ("dillenseger", "bamber", "branco"):
            raise ValueError("modelo_espalhadores deve ser 'dillenseger', 'bamber' ou 'branco'")

        self.f0_mhz = f0_mhz
        self.f0 = f0_mhz * 1e6
        self.c0 = c0_m_s
        self.lambda_mm = self.c0 / self.f0 * 1e3          # 0,308 mm a 5 MHz
        self.banda = banda_fracionaria                    # FWHM / f0 (-6 dB)
        self.f_number = f_number
        self.largura_sonda_mm = largura_sonda_mm
        self.profundidade_mm = profundidade_mm
        self.faixa_dinamica_db = faixa_dinamica_db
        self.tgc_db_cm_mhz = tgc_db_cm_mhz                # compensacao media (ida e volta)
        self.modelo_ecogenicidade = modelo_ecogenicidade
        self.modelo_espalhadores = modelo_espalhadores
        self.params_esp = {k: dict(v) for k, v in (params_espalhadores or PARAMS_ESPALHADORES).items()}
        self.semente = semente
        self.rng = np.random.default_rng(semente)          # reprodutibilidade

        # Grade fina de simulacao (mm) -- reamostragem exigida por Shannon [DLD sec. 2.3]
        self.dz_fino = self.lambda_mm / 8.0                # 4 amostras por periodo da portadora (lambda/2)
        self.dx_fino = self.lambda_mm / 4.0

        self._cache_hilbert = {}

    def reiniciar_rng(self):
        self.rng = np.random.default_rng(self.semente)

    # ------------------------------------------------------------------ #
    # Leitura
    # ------------------------------------------------------------------ #
    def carregar_serie_dicom(self, caminho_pasta: str):
        """
        Le a serie DICOM, reorienta para LPS e devolve (volume_hu[Z,Y,X], spacing(dx,dy,dz)).
        Apos a reorientacao: coluna -> esquerda do paciente, linha -> posterior, fatia -> superior.
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(caminho_pasta)
        if not dicom_names:
            raise FileNotFoundError(f"Nenhum arquivo DICOM foi encontrado em: {caminho_pasta}")
        reader.SetFileNames(dicom_names)
        imagem_sitk = reader.Execute()

        # Matriz de direcao do QIN-BREAST e (-1,-1,1): sem isso a linha 0 e POSTERIOR
        imagem_sitk = sitk.DICOMOrient(imagem_sitk, "LPS")

        volume_hu = sitk.GetArrayFromImage(imagem_sitk).astype(np.float32)
        # Pixels fora do FOV de reconstrucao vem como -3024 HU (padding) -> ar
        volume_hu = np.maximum(volume_hu, -1000.0)
        spacing = imagem_sitk.GetSpacing()  # (dx, dy, dz) em mm
        return volume_hu, spacing

    # ------------------------------------------------------------------ #
    # Mascara do corpo (para trocar o ar externo por agua)
    # ------------------------------------------------------------------ #
    @staticmethod
    def mascara_corpo(volume_hu: np.ndarray, limiar_hu: float = -400.0) -> np.ndarray:
        """
        Maior componente conexo com HU > limiar, com buracos (pulmoes) preenchidos fatia a fatia.
        O limiar deve ser IGUAL ao da classe "ar" em calcular_propriedades_acusticas.
        """
        rot, n = ndimage.label(volume_hu > limiar_hu)
        if n == 0:
            return np.zeros_like(volume_hu, dtype=bool)
        tamanhos = ndimage.sum_labels(np.ones_like(rot), rot, index=np.arange(1, n + 1))
        corpo = rot == (np.argmax(tamanhos) + 1)
        for k in range(corpo.shape[0]):
            corpo[k] = ndimage.binary_fill_holes(corpo[k])
        return corpo

    # ------------------------------------------------------------------ #
    # HU -> classes e propriedades acusticas  [DLD sec. 2.2, passo 1]
    # ------------------------------------------------------------------ #
    @staticmethod
    def classificar_tecidos(hu: np.ndarray, corpo: np.ndarray = None) -> np.ndarray:
        classe = np.full(hu.shape, GLANDULAR, dtype=np.int8)
        classe[hu < -400] = AR
        classe[(hu >= -400) & (hu < -30)] = GORDURA
        classe[hu > 150] = OSSO
        if corpo is not None:
            classe[~corpo] = AGUA
        return classe

    @staticmethod
    def calcular_propriedades_acusticas(hu: np.ndarray):
        """
        Mapeamento por classes de tecido. Impedancias de [DLD sec. 3.1]: ar 0,0004; gordura 1,35;
        tecido mole 1,65; osso 5 MRayl. Aqui, dentro de cada classe, ha variacao linear suave com o HU
        (extensao; fontes: Duck 1990; IT'IS v4.1; Mast 2000).
        """
        rho = np.empty_like(hu, dtype=np.float32)
        c = np.empty_like(hu, dtype=np.float32)
        alpha_0 = np.empty_like(hu, dtype=np.float32)    # dB/cm/MHz

        m_ar = hu < -400
        m_gord = (hu >= -400) & (hu < -30)
        m_gland = (hu >= -30) & (hu <= 150)
        m_osso = hu > 150

        rho[m_ar], c[m_ar], alpha_0[m_ar] = 1.2, 343.0, 12.0

        h = np.clip(hu[m_gord], -150.0, -30.0)
        rho[m_gord] = 925.0 + 0.25 * (h + 100.0)
        c[m_gord] = 1450.0 + 0.30 * (h + 100.0)
        alpha_0[m_gord] = 0.60

        h = hu[m_gland]
        rho[m_gland] = 1040.0 + 0.25 * (h - 40.0)
        c[m_gland] = 1540.0 + 0.40 * (h - 40.0)
        alpha_0[m_gland] = 0.75

        h = np.clip(hu[m_osso], 150.0, 1500.0)
        rho[m_osso] = 1100.0 + 0.60 * (h - 150.0)
        c[m_osso] = 1700.0 + 1.35 * (h - 150.0)
        alpha_0[m_osso] = 10.0

        Z = rho * c
        return rho, c, Z, alpha_0

    # ------------------------------------------------------------------ #
    # Espalhadores sub-resolucao
    # ------------------------------------------------------------------ #
    def _ordem_hilbert(self, shape):
        """Indices (achatados) dos pixels da janela na ordem da curva de Hilbert (com cache)."""
        if shape not in self._cache_hilbert:
            nz, nx = shape
            n = 1 << int(np.ceil(np.log2(max(nz, nx))))
            zz, xx = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
            d = hilbert_xy2d(n, xx.ravel(), zz.ravel())
            self._cache_hilbert[shape] = np.argsort(d, kind="stable")
        return self._cache_hilbert[shape]

    def _processo_gama(self, comprimento: int, p: float, alpha: float) -> np.ndarray:
        """
        Posicoes (em pixels ao longo da linha) de um processo de renovacao com distancias
        ~ Gama(alpha, beta), media d = alpha*beta = 1/p   [DLD Eq. 8].
        """
        d_media = 1.0 / p
        beta = d_media / alpha
        n_est = int(comprimento * p * 1.2 + 10 * np.sqrt(comprimento * p) + 10)
        pos = []
        inicio = self.rng.uniform(0.0, d_media)
        while True:
            passos = self.rng.gamma(alpha, beta, size=n_est)
            p_bloco = inicio + np.cumsum(passos)
            pos.append(p_bloco[p_bloco < comprimento])
            if p_bloco[-1] >= comprimento:
                break
            inicio = p_bloco[-1]
        return np.concatenate(pos)

    def campo_espalhadores(self, classe: np.ndarray) -> np.ndarray:
        """
        Campo adimensional de media ~0 e desvio ~1 por pixel (dentro de cada tecido) que
        modula a impedancia: Z_micro = Z * (1 + sigma_tecido * campo).
        """
        modelo = self.modelo_espalhadores
        campo = np.zeros(classe.shape, dtype=np.float32)

        if modelo == "branco":
            campo[:] = self.rng.standard_normal(classe.shape)
            return campo

        if modelo == "bamber":
            # Autocorrelacao gaussiana exp(-R^2/a^2) -> espectro de amplitude exp(-pi^2 a^2 s^2 / 2)
            # [B&D Eq. 7] -> nucleo espacial gaussiano com desvio a/2.
            ruido = self.rng.standard_normal(classe.shape).astype(np.float32)
            for c_id in np.unique(classe):
                prm = self.params_esp[int(c_id)]
                if prm["sigma"] == 0.0:
                    continue
                s = prm["a_corr_mm"] / 2.0
                f = ndimage.gaussian_filter(ruido, sigma=(s / self.dz_fino, s / self.dx_fino), mode="reflect")
                f /= (f.std() + 1e-12)
                m = classe == c_id
                campo[m] = f[m]
            return campo

        # modelo == "dillenseger": regularidade marcada + curva de Hilbert  [DLD sec. 2.2, passo 4]
        ordem = self._ordem_hilbert(classe.shape)
        classe_linha = classe.ravel()[ordem]
        L = classe_linha.size
        area_px = self.dz_fino * self.dx_fino                         # mm^2 por pixel
        var_ln = np.log(1.0 + 0.1)                                    # log-normal: media 1, var 0,1 [DLD]
        mu_ln = -0.5 * var_ln
        e_a2 = 1.0 + 0.1                                              # E[a^2]
        s_linha = np.zeros(L, dtype=np.float32)
        for c_id in np.unique(classe_linha):
            prm = self.params_esp[int(c_id)]
            p = prm["densidade_mm2"] * area_px                         # espalhadores por pixel
            if p <= 0.0 or prm["sigma"] == 0.0:
                continue
            pos = self._processo_gama(L, p, prm["alpha"])
            idx = pos.astype(np.int64)
            idx = idx[classe_linha[idx] == c_id]                       # mantem so os do tecido
            amp = self.rng.lognormal(mu_ln, np.sqrt(var_ln), size=idx.size).astype(np.float32)
            np.add.at(s_linha, idx, amp)
            m = classe_linha == c_id
            # centraliza e normaliza pelo equivalente de Poisson (preserva o efeito da regularidade)
            s_linha[m] = (s_linha[m] - p) / np.sqrt(p * e_a2)
        campo.ravel()[ordem] = s_linha
        return campo

    def sigma_por_classe(self, classe: np.ndarray) -> np.ndarray:
        tabela = np.zeros(max(self.params_esp) + 1, dtype=np.float32)
        for k, v in self.params_esp.items():
            tabela[k] = v["sigma"]
        return tabela[classe]

    # ------------------------------------------------------------------ #
    # PSF em unidades fisicas  [DLD Eq. 3]
    # ------------------------------------------------------------------ #
    def sigma_ax_dillenseger(self, Q: float = None) -> float:
        """sigma_ax de [DLD Eq. 9] (banda a -3 dB, coordenada de percurso), em mm."""
        Q = Q if Q is not None else 1.0 / self.banda
        return self.lambda_mm * Q * np.sqrt(np.log(2.0)) / np.pi

    def gerar_psf_transdutor(self, derivada2: bool = None):
        """
        PSF 2D (axial x lateral) amostrada na grade fina, com numero impar de amostras.
        Axial: pulso gaussiano de banda fracionaria B, ida e volta -> portadora cos(k z), k = 4*pi/lambda.
        Lateral: gaussiana com FWHM ~ lambda * F#.
        derivada2=True devolve d2H/dz2 / k^2 (analitico), usada em I = (d2H/dz2) * Z  [DLD sec. 3.1].
        A divisao por k^2 deixa a amplitude comparavel a do modelo "reflexao".
        """
        if derivada2 is None:
            derivada2 = self.modelo_ecogenicidade == "derivada2"
        sigma_t = 2.3548 / (2 * np.pi * self.banda * self.f0)     # s
        sigma_z = self.c0 * sigma_t / 2.0 * 1e3                   # mm (profundidade)
        sigma_x = self.lambda_mm * self.f_number / 2.3548         # mm
        k = 4 * np.pi / self.lambda_mm                            # rad/mm (ida e volta)

        nz = int(np.ceil(4 * sigma_z / self.dz_fino))
        nx = int(np.ceil(4 * sigma_x / self.dx_fino))
        z = np.arange(-nz, nz + 1) * self.dz_fino
        x = np.arange(-nx, nx + 1) * self.dx_fino
        zz, xx = np.meshgrid(z, x, indexing="ij")

        g = np.exp(-0.5 * (zz / sigma_z) ** 2)
        lat = np.exp(-0.5 * (xx / sigma_x) ** 2)
        cos_, sen_ = np.cos(k * zz), np.sin(k * zz)
        psf = g * cos_ * lat
        norma = np.sum(np.abs(psf))

        self.psf_info = dict(sigma_z_mm=sigma_z, sigma_x_mm=sigma_x,
                             fwhm_axial_mm=2.3548 * sigma_z, fwhm_lateral_mm=2.3548 * sigma_x,
                             shape=psf.shape, derivada2=derivada2)
        if not derivada2:
            return psf / norma

        # (g cos)'' = g'' cos + 2 g' cos' + g cos''
        g1 = -zz / sigma_z ** 2 * g
        g2 = (zz ** 2 / sigma_z ** 4 - 1.0 / sigma_z ** 2) * g
        d2 = (g2 * cos_ - 2.0 * k * g1 * sen_ - k ** 2 * g * cos_) * lat
        return d2 / (k ** 2) / norma

    # ------------------------------------------------------------------ #
    # Geometria da sonda
    # ------------------------------------------------------------------ #
    def posicionar_sonda(self, corpo: np.ndarray, spacing, centro_x_mm=None, folga_mm: float = 3.0):
        """
        Sonda linear paralela ao eixo L-R, feixe anterior->posterior, fixa ao longo das fatias.
        Por padrao centraliza no ponto mais anterior do corpo e posiciona a face 'folga_mm' antes da pele.
        """
        dx, dy, _ = spacing
        nz, ny, nx = corpo.shape
        proj = corpo.any(axis=0)                                  # (Y, X)
        primeira_linha = np.where(proj.any(axis=0), proj.argmax(axis=0), ny)

        if centro_x_mm is None:
            centro_x_mm = (np.argmin(primeira_linha) + 0.5) * dx
        meia = self.largura_sonda_mm / 2.0
        i0 = max(int((centro_x_mm - meia) / dx), 0)
        i1 = min(int(np.ceil((centro_x_mm + meia) / dx)), nx)
        pele_mm = primeira_linha[i0:i1].min() * dy
        topo_mm = max(pele_mm - folga_mm, 0.0)
        return centro_x_mm, topo_mm

    def coordenadas_janela(self, spacing, centro_x_mm, topo_mm):
        """Coordenadas (em indices da TC) da grade fina da janela da sonda."""
        dx, dy, _ = spacing
        nzf = int(round(self.profundidade_mm / self.dz_fino))
        nxf = int(round(self.largura_sonda_mm / self.dx_fino))
        prof = np.arange(nzf) * self.dz_fino
        lat = (np.arange(nxf) - (nxf - 1) / 2.0) * self.dx_fino
        lin = (topo_mm + prof) / dy - 0.5
        col = (centro_x_mm + lat) / dx - 0.5
        LL, CC = np.meshgrid(lin, col, indexing="ij")
        return np.stack([LL, CC]), prof, lat

    # ------------------------------------------------------------------ #
    # Simulacao de uma fatia
    # ------------------------------------------------------------------ #
    @staticmethod
    def _convolver_com_borda(campo: np.ndarray, psf: np.ndarray) -> np.ndarray:
        """Convolucao 'same' com extensao da borda (evita o degrau artificial no fundo da janela)."""
        pz, px = psf.shape[0] // 2, psf.shape[1] // 2
        estendido = np.pad(campo, ((pz, pz), (px, px)), mode="edge")
        return fftconvolve(estendido, psf, mode="valid")

    def simular_envelope_2d(self, hu_slice, corpo_slice, roi_slice, coords, psf):
        """
        Devolve o envelope (linear, sem normalizar) de uma fatia na janela da sonda.
        'psf' deve ser a gerada por gerar_psf_transdutor() com o mesmo modelo_ecogenicidade.
        """
        # Ar EXTERNO -> agua antes da interpolacao (ar interno/pulmao e mantido)
        hu_slice = np.where(corpo_slice, hu_slice, 0.0).astype(np.float32)

        # Reamostragem para a grade fina [DLD sec. 2.3]
        hu = ndimage.map_coordinates(hu_slice, coords, order=1, mode="nearest")
        corpo_suave = ndimage.gaussian_filter(corpo_slice.astype(np.float32), sigma=1.0)
        corpo = ndimage.map_coordinates(corpo_suave, coords, order=1, mode="constant") > 0.5
        peso_roi = ndimage.map_coordinates(roi_slice.astype(np.float32), coords, order=1, mode="constant", cval=0.0)

        _, _, Z, alpha_0 = self.calcular_propriedades_acusticas(hu)
        classe = self.classificar_tecidos(hu, corpo)

        # Fora do corpo / fora da ROI -> agua (transicao suave da apodizacao)
        w = peso_roi * corpo
        Z = Z * w + Z_AGUA * (1.0 - w)
        alpha_0 = alpha_0 * w + ALPHA_AGUA * (1.0 - w)

        # Espalhadores sub-resolucao
        sigma = self.sigma_por_classe(classe) * w
        Z_micro = Z * (1.0 + sigma * self.campo_espalhadores(classe))

        # Perda de transmissao nas interfaces (ida e volta) -- extensao, usa Z macroscopica
        R_macro = np.zeros_like(Z)
        R_macro[1:] = (Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1])
        T = np.cumprod(1.0 - R_macro ** 2, axis=0)
        T = np.vstack([np.ones((1, T.shape[1]), T.dtype), T[:-1]]) ** 2

        # Atenuacao ida e volta -- extensao
        dz_cm = self.dz_fino / 10.0
        atten_db = 2.0 * np.cumsum(alpha_0 * self.f0_mhz * dz_cm, axis=0)
        A = 10.0 ** (-atten_db / 20.0)

        if self.modelo_ecogenicidade == "derivada2":
            # [DLD Eq. 4-5]: I = H * (1/2Z0) d2Z/dz2 = (d2H/dz2) * (Z - Z0)/(2 Z0)
            # (Z0 e constante, entao subtrai-lo nao altera a derivada e evita artefato de borda)
            # T e A variam lentamente -> aplicados apos a convolucao.
            Z_rel = (Z_micro - Z_AGUA) / (2.0 * Z_AGUA)
            sinal_rf = self._convolver_com_borda(Z_rel, psf) * T * A
        else:
            # Versao anterior: refletividade R entre amostras consecutivas
            R_total = np.zeros_like(Z)
            R_total[1:] = (Z_micro[1:] - Z_micro[:-1]) / (Z_micro[1:] + Z_micro[:-1])
            sinal_rf = fftconvolve(R_total * T * A, psf, mode="same")

        envelope = np.abs(hilbert(sinal_rf, axis=0))               # [DLD Eq. 6]

        # TGC: compensa uma atenuacao media de referencia (dB/cm/MHz, ida e volta)
        prof_cm = np.arange(envelope.shape[0])[:, None] * dz_cm
        envelope *= 10.0 ** (2.0 * self.tgc_db_cm_mhz * self.f0_mhz * prof_cm / 20.0)
        return envelope.astype(np.float32)

    def comprimir_log(self, envelope_volume: np.ndarray, percentil: float = 99.5):
        """Normalizacao GLOBAL (mesma referencia para todas as fatias) + compressao log."""
        ref = np.percentile(envelope_volume[envelope_volume > 0], percentil)
        bmode = 20.0 * np.log10(np.maximum(envelope_volume / ref, 1e-12))
        return np.clip(bmode, -self.faixa_dinamica_db, 0.0).astype(np.float32)


# --- CLASSE PARA SELECAO DE ROI FREEHAND ---

class SelecionadorROIFreehand:
    """
    Interface grafica interativa para desenhar uma ROI freehand com o mouse.
    """
    def __init__(self, fatia_2d: np.ndarray):
        self.fatia_2d = fatia_2d
        self.mascara_2d = np.zeros_like(fatia_2d, dtype=bool)

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.imshow(fatia_2d, cmap='bone', vmin=-150, vmax=200)
        self.ax.set_title("DESENHE A ROI COM O MOUSE:\nClique, segure e contorne a mama. Solte para fechar.",
                          fontsize=11, fontweight='bold', color='darkred')
        self.ax.axis('off')

        self.lasso = LassoSelector(self.ax, onselect=self.onselect, props=dict(color='red', linewidth=2))
        plt.tight_layout()
        plt.show()

    def onselect(self, verts):
        ny, nx = self.fatia_2d.shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        pix = np.vstack((x.flatten(), y.flatten())).T
        self.mascara_2d = Path(verts).contains_points(pix).reshape((ny, nx))
        plt.close(self.fig)


def gerar_mascara_3d_a_partir_de_roi_freehand(volume_shape: tuple, mascara_2d: np.ndarray) -> np.ndarray:
    """
    Replica a mascara 2D em todas as fatias e suaviza a borda (apodizacao).
    """
    mascara_3d = np.broadcast_to(mascara_2d.astype(np.float32), volume_shape).copy()
    return ndimage.gaussian_filter(mascara_3d, sigma=(0, 2.0, 2.0))


# --- COMPARACAO ENTRE MODELOS ---

def comparar_modelos(pipeline_base: PipelineTCparaUltrassom, hu_slice, corpo_slice, roi_slice,
                     coords, extent_us):
    """
    Simula a MESMA fatia com todas as combinacoes (ecogenicidade x espalhadores).
    Cada imagem e normalizada individualmente (so para comparar textura).
    """
    ecos = ["reflexao", "derivada2"]
    esps = ["branco", "bamber", "dillenseger"]
    fig, axes = plt.subplots(len(ecos), len(esps), figsize=(5 * len(esps), 5.5 * len(ecos)), squeeze=False)
    for i, eco in enumerate(ecos):
        for j, esp in enumerate(esps):
            p = PipelineTCparaUltrassom(
                f0_mhz=pipeline_base.f0_mhz, c0_m_s=pipeline_base.c0,
                banda_fracionaria=pipeline_base.banda, f_number=pipeline_base.f_number,
                largura_sonda_mm=pipeline_base.largura_sonda_mm, profundidade_mm=pipeline_base.profundidade_mm,
                faixa_dinamica_db=pipeline_base.faixa_dinamica_db, tgc_db_cm_mhz=pipeline_base.tgc_db_cm_mhz,
                modelo_ecogenicidade=eco, modelo_espalhadores=esp,
                params_espalhadores=pipeline_base.params_esp, semente=pipeline_base.semente)
            psf = p.gerar_psf_transdutor()
            env = p.simular_envelope_2d(hu_slice, corpo_slice, roi_slice, coords, psf)[::2]
            img = p.comprimir_log(env)
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=-p.faixa_dinamica_db, vmax=0, extent=extent_us, aspect="equal")
            marca = "  (referencias)" if (eco == "derivada2" and esp != "branco") else ""
            marca = "  (versao anterior)" if (eco == "reflexao" and esp == "branco") else marca
            ax.set_title(f"{eco} + {esp}{marca}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Lateral (mm)")
            if j == 0:
                ax.set_ylabel("Profundidade (mm)")
    plt.tight_layout()
    plt.show()


# --- FUNCOES DE VISUALIZACAO INTERATIVA ---

def visualizar_fatiador_interativo(volume_3d: np.ndarray, extent, faixa_db: float):
    fig, ax = plt.subplots(figsize=(7, 8))
    plt.subplots_adjust(bottom=0.2)
    k0 = volume_3d.shape[0] // 2
    img = ax.imshow(volume_3d[k0], cmap='gray', vmin=-faixa_db, vmax=0, extent=extent)
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Profundidade (mm)")
    ax.set_title(f"Fatia {k0 + 1}/{volume_3d.shape[0]}")

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Fatia', 0, volume_3d.shape[0] - 1, valinit=k0, valfmt='%d', valstep=1)

    def update(val):
        idx = int(slider.val)
        img.set_data(volume_3d[idx])
        ax.set_title(f"Fatia {idx + 1}/{volume_3d.shape[0]}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def visualizar_volume_3d_pyvista(volume_3d: np.ndarray, spacing_zyx: tuple, faixa_db: float):
    """
    volume_3d em (fatia, profundidade, lateral); spacing_zyx no mesmo ordem (mm).
    """
    if pv is None:
        print("PyVista nao instalado; pulando renderizacao 3D.")
        return
    grid = pv.ImageData()
    grid.dimensions = np.array(volume_3d.shape) + 1
    grid.spacing = spacing_zyx
    grid.cell_data["Ultrassom_BMode"] = volume_3d.flatten(order="F")

    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars="Ultrassom_BMode", cmap="gray",
                       opacity=[0.0, 0.0, 0.2, 0.6, 0.9], clim=[-faixa_db, 0])
    plotter.add_axes(xlabel="Fatia (S-I)", ylabel="Profundidade", zlabel="Lateral")
    plotter.show_grid()
    plotter.show(title="Volume 3D Ultrassonico")


# --- MAIN ---
if __name__ == "__main__":
    caminho_serie_dicom = r"C:\Users\TOMMUS\Downloads\QIN-BREAST_2015-09-04\qin_breast\QIN-BREAST-01-0002\1992-01-24-BREAST PRONE-12000\2-CTAC-14691"

    # ---------------- CONFIGURACAO DO MODELO ----------------
    MODELO_ECOGENICIDADE = "derivada2"     # "derivada2" [DLD Eq.4] | "reflexao" (versao anterior)
    MODELO_ESPALHADORES = "dillenseger"    # "dillenseger" [DLD] | "bamber" [B&D] | "branco" (versao anterior)
    COMPARAR_MODELOS = True                # painel 2x3 com todas as combinacoes na fatia central
    # --------------------------------------------------------

    print("Inicializando a Conversao de TC para ultrassom...")
    pipeline = PipelineTCparaUltrassom(f0_mhz=5,
                                       modelo_ecogenicidade=MODELO_ECOGENICIDADE,
                                       modelo_espalhadores=MODELO_ESPALHADORES)
    print(f"   -> Modelo: ecogenicidade = {MODELO_ECOGENICIDADE}, espalhadores = {MODELO_ESPALHADORES}")

    if not os.path.exists(caminho_serie_dicom):
        raise SystemExit(f"Erro: O caminho especificado nao existe:\n{caminho_serie_dicom}")

    print("1. Carregando e reorientando o volume DICOM (LPS)...")
    volume_hu, spacing = pipeline.carregar_serie_dicom(caminho_serie_dicom)
    num_fatias = volume_hu.shape[0]
    print(f"   -> Volume: {volume_hu.shape} (Fatias x Linhas x Colunas)")
    print(f"   -> Voxel (dx, dy, dz): {spacing} mm | lambda = {pipeline.lambda_mm:.3f} mm")

    print("2. Segmentando o corpo (ar externo -> agua)...")
    corpo = pipeline.mascara_corpo(volume_hu)

    print("\n==============================================")
    print(" ESCOLHA O MODO DE PROCESSAMENTO:")
    print(" [1] Desenhar ROI Freehand (Desenho com o mouse)")
    print(" [2] Usar Imagem Inteira (Sem ROI)")
    print("==============================================")
    opcao = input("Digite 1 ou 2 e pressione Enter: ").strip()

    mascara_2d = None
    if opcao == "1":
        seletor_roi = SelecionadorROIFreehand(volume_hu[num_fatias // 2])
        if np.any(seletor_roi.mascara_2d):
            mascara_2d = seletor_roi.mascara_2d
            print("   -> ROI Freehand capturada.")
        else:
            print("   -> Nenhuma ROI desenhada. Processando imagem inteira.")
    if mascara_2d is not None:
        mascara_roi_3d = gerar_mascara_3d_a_partir_de_roi_freehand(volume_hu.shape, mascara_2d)
        corpo_sonda = corpo & (mascara_roi_3d > 0.5)
    else:
        mascara_roi_3d = np.ones_like(volume_hu, dtype=np.float32)
        corpo_sonda = corpo

    print("3. Posicionando a sonda e gerando a PSF...")
    centro_x_mm, topo_mm = pipeline.posicionar_sonda(corpo_sonda, spacing)
    coords, prof, lat = pipeline.coordenadas_janela(spacing, centro_x_mm, topo_mm)
    psf = pipeline.gerar_psf_transdutor()
    print(f"   -> Sonda: centro lateral {centro_x_mm:.1f} mm, face em y = {topo_mm:.1f} mm")
    print(f"   -> PSF: FWHM axial {pipeline.psf_info['fwhm_axial_mm']:.2f} mm, "
          f"lateral {pipeline.psf_info['fwhm_lateral_mm']:.2f} mm, {psf.shape} amostras")
    print(f"   -> sigma axial (conv. deste codigo) = {pipeline.psf_info['sigma_z_mm']:.3f} mm | "
          f"sigma_ax [DLD Eq. 9] com Q = 1/B = {pipeline.sigma_ax_dillenseger():.3f} mm")

    extent_us = [lat[0], lat[-1], prof[-1], prof[0]]
    k = num_fatias // 2

    if COMPARAR_MODELOS:
        print("   -> Gerando painel comparativo dos modelos na fatia central...")
        comparar_modelos(pipeline, volume_hu[k], corpo[k], mascara_roi_3d[k], coords, extent_us)

    print("4. Simulando o volume fatia por fatia...")
    t0 = time.time()
    envelopes = np.empty((num_fatias,) + coords.shape[1:], dtype=np.float32)
    for kk in range(num_fatias):
        envelopes[kk] = pipeline.simular_envelope_2d(volume_hu[kk], corpo[kk], mascara_roi_3d[kk], coords, psf)
    print(f"   -> {num_fatias} fatias em {time.time() - t0:.1f} s")

    # Decima o eixo axial por 2 (envelope tem banda bem menor que a portadora)
    envelopes = envelopes[:, ::2, :]
    volume_ultrassom_3d = pipeline.comprimir_log(envelopes)
    dz_out = pipeline.dz_fino * 2
    print(f"   -> Volume concluido: {volume_ultrassom_3d.shape}, "
          f"voxel (fatia, prof, lat) = ({spacing[2]:.2f}, {dz_out:.3f}, {pipeline.dx_fino:.3f}) mm")

    # --- FIGURA COMPARATIVA ---
    dx, dy, _ = spacing
    ny, nx = volume_hu.shape[1:]
    extent_ct = [0, nx * dx, ny * dy, 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    im0 = axes[0].imshow(volume_hu[k], cmap='bone', vmin=-150, vmax=200, extent=extent_ct)
    x0, x1 = centro_x_mm - pipeline.largura_sonda_mm / 2, centro_x_mm + pipeline.largura_sonda_mm / 2
    axes[0].add_patch(plt.Rectangle((x0, topo_mm), x1 - x0, pipeline.profundidade_mm,
                                    fill=False, ec='yellow', lw=1.5))
    if mascara_2d is not None:
        axes[0].contour(mascara_2d, levels=[0.5], colors='red', linewidths=1, extent=[0, nx * dx, 0, ny * dy])
    axes[0].set_title("1. TC (LPS, anterior no topo) + janela da sonda", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("x (mm)"); axes[0].set_ylabel("y (mm)")
    fig.colorbar(im0, ax=axes[0], label="HU")

    hu_win = ndimage.map_coordinates(volume_hu[k], coords, order=1, mode="nearest")
    im1 = axes[1].imshow(hu_win, cmap='bone', vmin=-150, vmax=200, extent=extent_us, aspect='equal')
    axes[1].set_title("2. TC reamostrada na janela da sonda", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Lateral (mm)"); axes[1].set_ylabel("Profundidade (mm)")
    fig.colorbar(im1, ax=axes[1], label="HU")

    im2 = axes[2].imshow(volume_ultrassom_3d[k], cmap='gray', vmin=-pipeline.faixa_dinamica_db, vmax=0,
                         extent=extent_us, aspect='equal')
    axes[2].set_title(f"3. US simulado ({MODELO_ECOGENICIDADE} + {MODELO_ESPALHADORES})",
                      fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Lateral (mm)")
    fig.colorbar(im2, ax=axes[2], label="dB")
    plt.tight_layout()
    plt.show()

    visualizar_fatiador_interativo(volume_ultrassom_3d, extent_us, pipeline.faixa_dinamica_db)
    visualizar_volume_3d_pyvista(volume_ultrassom_3d, (spacing[2], dz_out, pipeline.dx_fino),
                                 pipeline.faixa_dinamica_db)
