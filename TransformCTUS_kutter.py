"""
Conversao CT -> Ultrassom seguindo o modelo de Kutter, Shams & Navab (2009)

Referencia do banco de dados:
Li, X., Abramson, R. G., Arlinghaus, L. R., Chakravarthy, A. B., Abramson, V. G., Sanders, M., & Yankeelov, T. E. (2016).
Data From QIN-BREAST (Version 2) [Data set].
The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.21JUEBH0

Referencia do modelo:
[KSN] Kutter, O., Shams, R., & Navab, N. (2009). Visualization and GPU-accelerated simulation of
      medical ultrasound from CT images. Comput. Methods Programs Biomed., 94(3), 250-266.
      https://doi.org/10.1016/j.cmpb.2008.12.011
      (extensao de Shams, Hartley & Navab, MICCAI 2008)

Instalar:
pip install SimpleITK scipy numpy matplotlib pyvista

Modelo implementado (conferido com o PDF de [KSN], secoes 2.1 a 2.3 e Fig. 10)
-------------------------------------------------------------------------------
O feixe propaga no sentido anterior -> posterior (linhas da TC em LPS), uma linha de varredura
por coluna da grade fina. A regiao simulada e a ROI desenhada no inicio (ou o corpo inteiro).

1. Impedancia [KSN, sec. 2.1]: relacao aproximadamente LINEAR entre HU e Z para tecido mole
   (Wein et al. 2007); ar e osso sao SEGMENTADOS por limiares e recebem impedancia propria,
   "onde os valores de TC nao podem ser usados diretamente":
       tecido mole: Z = Z_agua * (HU + 1000) / 1000
       HU < limiar_ar_hu  -> Z_ar ;  HU > limiar_osso_hu -> Z_osso
   Os limiares da Fig. 10 sao 0,044 (ar) e 0,313 (osso) em TC normalizada; supondo a janela de
   12 bits [-1024, 3071] HU, equivalem a ~ -844 HU e ~ +258 HU (suposicao: a normalizacao nao
   e descrita no artigo).

2. Coeficientes de intensidade [KSN, Eq. 1]:  alpha_R = ((Z2 - Z1)/(Z2 + Z1))^2,  alpha_T = 1 - alpha_R.
   Aqui alpha_R e calculado com a diferenca de Z na escala de um voxel da TC, mantendo so o
   maximo local de cada interface ao longo do feixe (uma reflexao por interface).

3. Intensidade incidente, recursiva [KSN, Eq. 5]:  I_i(x) = I_i(x - d) * (1 - alpha_R(x - d)).
   O artigo considera SO perdas de transmissao (sem absorcao). Isso da uma sombra de apenas
   ~ -11 dB sob uma costela; por isso o padrao absorcao="osso" acrescenta absorcao so no osso e
   no ar segmentados (extensao). absorcao="nenhuma" reproduz a Eq. 5 literal; "todas" aplica
   absorcao em todos os tecidos (use com tgc_db_cm_mhz > 0).

4. Reflexao lambertiana [KSN, Eq. 2-3]:  I_r(x) ~ alpha_R(x) * I_i(x)^2 / I_0 * |r(x).n(x)|
   n = normal da superficie (aqui: gradiente 3D de Deriche; o artigo nao diz como n e obtido).

5. Abertura ativa [KSN, Eq. 4]: integral de x - l a x + l com janela w (quadrada, triangular ou
   Hann), l = n_a (w_e + s_e) = n_a * pitch. Fig. 10: Hann, 13 elementos na reflexao e 12 no
   espalhamento. A mesma janela "alonga" o speckle perpendicularmente ao feixe [KSN, sec. 2.3].
   Padrao: largura total n_a * pitch (janela_literal=True usa 2 n_a * pitch, como na Eq. 4).

6. Imagem de espalhamento I_s: [KSN] usa o Field II sobre a TC pre-processada "como descrito em"
   Shams et al. 2008 (o artigo de 2009 nao detalha esse pre-processamento). Aqui: mapa m da TC
   suavizada (estiramento afim, <= 5% saturados, bordas de Canny/Deriche = 1, piso de
   ecogenicidade), espalhadores Poisson com amplitude de media zero, convolucao com a PSF,
   envelope^2 -> I_s (media 1 onde m = 1), depois janela de Hann lateral (alongamento).

7. Combinacao [KSN, Eq. 6]:
       I_us(x) = ( G_sigma1(x) * I_r(x) + alpha * G_sigma2(x) * alpha_T(x) ) * I_s(x)
   G = filtros gaussianos (sigma1 na reflexao, sigma2 na transmissao); alpha = mistura.
   alpha_T(x) aqui = transmissao acumulada I_i(x)/I_0 (o "transmission texture" das Figs. 5-6),
   que impede speckle nas sombras. Fig. 10: alpha = 0,0009, sigma_R = 1,8 e sigma_T = 4,8 pixels
   (convertidos aqui para mm supondo 0,3 mm por pixel: 0,55 e 1,45 mm).

8. Compressao log [KSN, Eq. 7]:
       I_c = 10 log10( I_us / max(I_us) ) + beta,  e 0 onde I_us < max * 10^(-beta/10)
   beta = faixa dinamica (Fig. 10 usa ~103 dB; aqui 50 dB por padrao). Saida em [0, beta] dB;
   comprimir_log devolve [-beta, 0] (mesma imagem, deslocada).
"""

import os
import time
import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import hilbert, fftconvolve, lfilter
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, LassoSelector
from matplotlib.path import Path
import SimpleITK as sitk

try:
    import pyvista as pv
except ImportError:
    pv = None


RHO_AGUA, C_AGUA = 1000.0, 1480.0
Z_AGUA = RHO_AGUA * C_AGUA
Z_AR = 1.2 * 343.0
ALPHA_AGUA = 0.0022          # dB/cm/MHz


# ---------------------------------------------------------------------- #
# Filtro de Deriche (recursivo) -- gradiente 3D para as normais de superficie
# ---------------------------------------------------------------------- #
def _deriche_1d(x: np.ndarray, alpha: float, axis: int, derivada: bool) -> np.ndarray:
    """
    Filtro de Deriche 1D (suavizacao ou derivada) ao longo de 'axis'.
    Coeficientes de Deriche (1987); ganho normalizado numericamente (constante -> 1 na
    suavizacao, rampa unitaria -> 1 na derivada), em unidades de 1/amostra.
    """
    ea, e2a = np.exp(-alpha), np.exp(-2.0 * alpha)
    b1, b2 = 2.0 * ea, -e2a
    if derivada:
        a1, a2, a3, a4 = 0.0, 1.0, -1.0, 0.0
    else:
        k = (1.0 - ea) ** 2 / (1.0 + 2.0 * alpha * ea - e2a)
        a1, a2, a3, a4 = k, k * ea * (alpha - 1.0), k * ea * (alpha + 1.0), -k * e2a

    def aplicar(v, ax):
        y1 = lfilter([a1, a2], [1.0, -b1, -b2], v, axis=ax)
        y2 = np.flip(lfilter([0.0, a3, a4], [1.0, -b1, -b2], np.flip(v, axis=ax), axis=ax), axis=ax)
        return y1 + y2

    # ganho (calculado num sinal de teste longo, longe das bordas)
    n_teste = 200
    teste = np.arange(n_teste, dtype=np.float64) if derivada else np.ones(n_teste)
    ganho = aplicar(teste, 0)[n_teste // 2]

    pad = int(np.ceil(8.0 / alpha)) + 2
    larg = [(0, 0)] * x.ndim
    larg[axis] = (pad, pad)
    xp = np.pad(x, larg, mode="edge")
    y = aplicar(xp, axis) / ganho
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(pad, pad + x.shape[axis])
    return y[tuple(sl)]


def gradiente_deriche_3d(vol: np.ndarray, spacing_zyx, escala_mm: float = 1.5):
    """
    Gradiente 3D (por mm) com filtros de Deriche: derivada num eixo, suavizacao nos outros.
    alpha por eixo = espacamento / escala_mm (mesma escala fisica em todos os eixos).
    Devolve (gz, gy, gx).
    """
    vol = vol.astype(np.float64)
    alphas = [s / escala_mm for s in spacing_zyx]
    grads = []
    for eixo in range(3):
        g = vol
        for outro in range(3):
            g = _deriche_1d(g, alphas[outro], outro, derivada=(outro == eixo))
        grads.append((g / spacing_zyx[eixo]).astype(np.float32))
    return tuple(grads)


class SimuladorKutter:
    """
    Simulacao de ultrassom a partir da TC pelo modelo de raios de [KSN].
    """

    def __init__(self, f0_mhz: float = 5.0, c0_m_s: float = 1540.0,
                 banda_fracionaria: float = 0.6, f_number: float = 2.0,
                 alpha_mistura: float = 0.0009, expoente_lambert: float = 1.0,
                 limiar_ar_hu: float = -844.0, limiar_osso_hu: float = 258.0,
                 z_osso: float = 7.8e6,
                 pitch_mm: float = 0.2875, elementos_ativos_reflexao: int = 13,
                 elementos_ativos_espalhamento: int = 12, janela: str = "hann",
                 sigma_reflexao_mm: float = 0.55, sigma_transmissao_mm: float = 1.45,
                 absorcao: str = "osso", janela_literal: bool = False,
                 normalizar_pico_reflexao: bool = True, suavizacao_ct_z_mm: float = 1.5,
                 saturacao_estiramento: float = 0.05,
                 suavizacao_ct_mm: float = 1.5,
                 modo_bordas: str = "canny", limiar_borda_alto: float = 15.0,
                 limiar_borda_baixo: float = 8.0, percentil_bordas: float = 95.0,
                 valor_borda: float = 1.0,
                 ecogenicidade_minima: float = 0.05,
                 escala_deriche_mm: float = 1.5,
                 espalhadores_por_celula: float = 10.0, modelo_amplitude: str = "media_zero",
                 desvio_amplitude: float = 0.1,
                 tgc_db_cm_mhz: float = 0.0, faixa_dinamica_db: float = 35.0,
                 referencia_log: str = "max",
                 semente: int = 0):
        self.f0_mhz = f0_mhz
        self.f0 = f0_mhz * 1e6
        self.c0 = c0_m_s
        self.lambda_mm = self.c0 / self.f0 * 1e3
        self.banda = banda_fracionaria
        self.f_number = f_number
        self.alpha_mistura = alpha_mistura            # alpha da Eq. 6 (Fig. 10: 0,0009)
        self.limiar_ar_hu = limiar_ar_hu
        self.limiar_osso_hu = limiar_osso_hu
        self.z_osso = z_osso                          # osso cortical ~7,4-7,8 MRayl
        self.pitch_mm = pitch_mm                      # w_e + s_e (L12-5: 36,8 mm / 128 elementos, suposicao)
        self.na_reflexao = elementos_ativos_reflexao
        self.na_espalhamento = elementos_ativos_espalhamento
        self.janela = janela                          # "hann", "triangular" ou "quadrada"
        self.sigma_reflexao_mm = sigma_reflexao_mm
        self.sigma_transmissao_mm = sigma_transmissao_mm
        self.absorcao = absorcao                      # "nenhuma" (Eq. 5 literal), "osso" (padrao) ou "todas"
        self.janela_literal = janela_literal
        # G_s1 com pico unitario: o artigo da sigma_R = 1,8 pixel da imagem DELE; convertido para mm
        # na nossa grade fina (~14 amostras), um filtro de soma 1 dividia o pico de cada interface
        # por ~36 (-15,5 dB) e as reflexoes sumiam sob o speckle (ver Diagnostico_etapas.py).
        self.normalizar_pico_reflexao = normalizar_pico_reflexao
        # TC suavizada (no plano) so para calcular Z: evita reflexoes geradas pelo ruido da CTAC
        self.suavizacao_ct_z_mm = suavizacao_ct_z_mm
        self.referencia_log = referencia_log          # "max" (Eq. 7) ou "percentil"
        self.p_lambert = expoente_lambert
        self.saturacao = saturacao_estiramento
        self.percentil_bordas = percentil_bordas
        self.suavizacao_ct_mm = suavizacao_ct_mm
        self.modo_bordas = modo_bordas                    # "canny" (padrao) ou "percentil" (versao anterior)
        self.limiar_borda_alto = limiar_borda_alto        # HU/mm
        self.limiar_borda_baixo = limiar_borda_baixo      # HU/mm
        self.valor_borda = valor_borda
        self.ecogenicidade_minima = ecogenicidade_minima  # piso de m no tecido (0 = estiramento puro de [KSN])
        self.modelo_amplitude = modelo_amplitude          # "media_zero" (padrao) ou "kutter"
        self.escala_deriche_mm = escala_deriche_mm
        self.esp_por_celula = espalhadores_por_celula
        self.desvio_amp = desvio_amplitude
        self.tgc_db_cm_mhz = tgc_db_cm_mhz
        self.faixa_dinamica_db = faixa_dinamica_db
        self.rng = np.random.default_rng(semente)

        # Grade fina (Shannon): lambda/8 axial, lambda/4 lateral
        self.dz_fino = self.lambda_mm / 8.0
        self.dx_fino = self.lambda_mm / 4.0

        # PSF (mesma convencao dos scripts anteriores)
        sigma_t = 2.3548 / (2 * np.pi * self.banda * self.f0)
        self.sigma_z = self.c0 * sigma_t / 2.0 * 1e3            # mm, envelope de amplitude
        self.sigma_x = self.lambda_mm * self.f_number / 2.3548  # mm
        self.fwhm_lateral = 2.3548 * self.sigma_x
        self.psf = self._gerar_psf()
        self._norma_is = None

    # ------------------------------------------------------------------ #
    # Leitura e mascaras
    # ------------------------------------------------------------------ #
    @staticmethod
    def carregar_serie_dicom(caminho_pasta: str):
        reader = sitk.ImageSeriesReader()
        nomes = reader.GetGDCMSeriesFileNames(caminho_pasta)
        if not nomes:
            raise FileNotFoundError(f"Nenhum arquivo DICOM foi encontrado em: {caminho_pasta}")
        reader.SetFileNames(nomes)
        img = sitk.DICOMOrient(reader.Execute(), "LPS")        # linha 0 = anterior
        vol = np.maximum(sitk.GetArrayFromImage(img).astype(np.float32), -1000.0)
        return vol, img.GetSpacing()                           # (dx, dy, dz)

    @staticmethod
    def mascara_corpo(volume_hu: np.ndarray, limiar_hu: float = -400.0) -> np.ndarray:
        rot, n = ndimage.label(volume_hu > limiar_hu)
        if n == 0:
            return np.zeros_like(volume_hu, dtype=bool)
        tam = ndimage.sum_labels(np.ones_like(rot), rot, index=np.arange(1, n + 1))
        corpo = rot == (np.argmax(tam) + 1)
        for k in range(corpo.shape[0]):
            corpo[k] = ndimage.binary_fill_holes(corpo[k])
        return corpo

    @staticmethod
    def regiao_da_roi(corpo, spacing, mascara_2d=None, folga_mm=3.0, margem_mm=2.0):
        """Retangulo envolvente da ROI (ou do corpo), topo numa camada de agua acima da pele."""
        dx, dy, _ = spacing
        ny, nx = corpo.shape[1:]
        proj = corpo.any(axis=0)
        regiao = proj if mascara_2d is None else mascara_2d.astype(bool)
        tecido = regiao & proj
        if not tecido.any():
            raise ValueError("A ROI nao contem tecido do corpo.")
        linhas = np.where(regiao.any(axis=1))[0]
        cols = np.where(regiao.any(axis=0))[0]
        pele = np.where(tecido.any(axis=1))[0][0]
        x0 = max(cols[0] * dx - margem_mm, 0.0)
        x1 = min((cols[-1] + 1) * dx + margem_mm, nx * dx)
        topo = max(pele * dy - folga_mm, 0.0)
        fundo = min((linhas[-1] + 1) * dy + margem_mm, ny * dy)
        return x0, x1, topo, fundo

    def coordenadas_regiao(self, spacing, x0, x1, topo, fundo):
        dx, dy, _ = spacing
        nzf = int(round((fundo - topo) / self.dz_fino))
        nxf = int(round((x1 - x0) / self.dx_fino))
        prof = np.arange(nzf) * self.dz_fino
        xs = x0 + (np.arange(nxf) + 0.5) * self.dx_fino
        LL, CC = np.meshgrid((topo + prof) / dy - 0.5, xs / dx - 0.5, indexing="ij")
        return np.stack([LL, CC]), prof, xs

    # ------------------------------------------------------------------ #
    # Pre-processamento do volume (uma vez): normais e mapa de espalhamento
    # ------------------------------------------------------------------ #
    def preprocessar_volume(self, volume_hu, corpo, spacing, roi_3d=None):
        """
        Calcula no volume da TC:
          - |r . n|^p : r = +y (anterior -> posterior), n = gradiente de Deriche normalizado
          - mapa de espalhamento m em [0, 1]: estiramento afim (<= 5% saturados) + bordas = 1
        """
        dx, dy, dz = spacing
        hu = np.where(corpo, volume_hu, 0.0).astype(np.float32)   # ar externo -> agua
        gz, gy, gx = gradiente_deriche_3d(hu, (dz, dy, dx), self.escala_deriche_mm)
        gmag = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)
        rn = (np.abs(gy) / (gmag + 1e-6)) ** self.p_lambert

        regiao = corpo if roi_3d is None else (corpo & (roi_3d > 0.5))

        # (a) A CTAC e de baixa dose (ruido ~15-30 HU). Sem suavizar, o ruido vira manchas de
        #     ecogenicidade do tamanho do voxel (1,37 mm) por cima do speckle.
        if self.suavizacao_ct_mm > 0:
            s = self.suavizacao_ct_mm
            hu_m = ndimage.gaussian_filter(hu, sigma=(s / dz, s / dy, s / dx), mode="nearest")
        else:
            hu_m = hu
        vals = hu_m[regiao]
        lo, hi = np.percentile(vals, [100 * self.saturacao / 2, 100 * (1 - self.saturacao / 2)])
        m = np.clip((hu_m - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
        # (c) Piso de ecogenicidade no tecido: na mama a gordura ocupa a faixa baixa do
        #     estiramento e iria para m ~ 0 (anecoica, buracos pretos). Gordura e hipoecoica,
        #     nao anecoica: m_tecido = m_min + (1 - m_min) * m. Com m_min = 0,2 a gordura fica
        #     ~14 dB abaixo da glandula (20*log10(0,2)); com 0,3, ~10 dB.
        m = self.ecogenicidade_minima + (1.0 - self.ecogenicidade_minima) * m

        # (b) Bordas finas (Canny sobre o gradiente de Deriche): supressao de nao-maximos +
        #     histerese com limiares FISICOS (HU/mm). Um percentil (ex.: 5% dos voxels) marcava
        #     como borda faixas grossas e, numa CTAC ruidosa, pontos de ruido espalhados.
        if self.modo_bordas == "canny":
            bordas = self._bordas_canny(gmag, gy, gx) & corpo
        else:
            bordas = (gmag > np.percentile(gmag[regiao], self.percentil_bordas)) & corpo
        m[bordas] = np.maximum(m[bordas], self.valor_borda)
        m[~corpo] = 0.0
        self.info_preproc = dict(hu_lo=lo, hu_hi=hi, frac_bordas=bordas[regiao].mean())
        return rn.astype(np.float32), m, gmag

    def _bordas_canny(self, gmag, gy, gx):
        """Supressao de nao-maximos no plano (direcao do gradiente) + histerese, fatia a fatia."""
        ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
        setor = (((ang + 22.5) // 45.0) % 4).astype(np.int8)       # 0: x, 1: diag, 2: y, 3: anti-diag
        desloc = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}
        maximo = np.zeros_like(gmag, dtype=bool)
        for s_id, (dyy, dxx) in desloc.items():
            frente = np.roll(gmag, (-dyy, -dxx), axis=(1, 2))
            tras = np.roll(gmag, (dyy, dxx), axis=(1, 2))
            maximo |= (setor == s_id) & (gmag >= frente) & (gmag >= tras)
        forte = maximo & (gmag >= self.limiar_borda_alto)
        fraco = maximo & (gmag >= self.limiar_borda_baixo)
        rot, n = ndimage.label(fraco, structure=np.ones((3, 3, 3)))
        if n == 0:
            return forte
        manter = np.zeros(n + 1, dtype=bool)
        manter[np.unique(rot[forte])] = True
        manter[0] = False
        return manter[rot]

    # ------------------------------------------------------------------ #
    # PSF e normalizacao de I_s
    # ------------------------------------------------------------------ #
    def _gerar_psf(self):
        k = 4 * np.pi / self.lambda_mm
        nz = int(np.ceil(4 * self.sigma_z / self.dz_fino))
        nx = int(np.ceil(4 * self.sigma_x / self.dx_fino))
        z = np.arange(-nz, nz + 1) * self.dz_fino
        x = np.arange(-nx, nx + 1) * self.dx_fino
        zz, xx = np.meshgrid(z, x, indexing="ij")
        psf = np.exp(-0.5 * ((zz / self.sigma_z) ** 2 + (xx / self.sigma_x) ** 2)) * np.cos(k * zz)
        return psf / np.sum(np.abs(psf))

    def _espalhamento_rf(self, m):
        """
        Espalhadores em posicoes aleatorias (Poisson na grade fina) convoluidos com a PSF.
        - "media_zero" (padrao, como nos fantomas do Field II): cada espalhador tem amplitude
          ~ N(0, m^2); a ecogenicidade (intensidade media) e proporcional a densidade * m^2.
        - "kutter": amplitude ~ N(m, (s*m)^2) (media dependente da posicao, como no resumo de [KSN]).
        Densidade: espalhadores_por_celula na celula de resolucao (>= ~10 para speckle
        totalmente desenvolvido; abaixo disso o speckle fica pontilhado e nao-Rayleigh).
        """
        celula = (2.3548 * self.sigma_z) * self.fwhm_lateral          # mm^2 (celula de resolucao)
        p = self.esp_por_celula * self.dz_fino * self.dx_fino / celula
        kk = self.rng.poisson(p, size=m.shape).astype(np.float32)
        xi = self.rng.standard_normal(m.shape).astype(np.float32)
        if self.modelo_amplitude == "media_zero":
            amp = m * np.sqrt(kk) * xi                                # soma de kk termos N(0, m^2)
        else:
            amp = m * (kk + np.sqrt(kk) * self.desvio_amp * xi)
        return fftconvolve(amp, self.psf, mode="same")

    def _normalizacao_is(self):
        if self._norma_is is None:
            rf = self._espalhamento_rf(np.ones((600, 400), np.float32))
            env2 = np.abs(hilbert(rf, axis=0)) ** 2
            self._norma_is = float(env2[50:-50, 50:-50].mean())
        return self._norma_is

    # ------------------------------------------------------------------ #
    # Simulacao de uma fatia
    # ------------------------------------------------------------------ #
    @staticmethod
    def alpha_por_hu(hu):
        """Atenuacao (dB/cm/MHz) por faixa de HU (valores de literatura; ajustaveis)."""
        a = np.full(hu.shape, 0.75, np.float32)        # tecido mole / glandula
        a[hu < -30] = 0.60                              # gordura
        a[hu < -400] = 12.0                             # ar (pulmao)
        a[hu > 150] = 10.0                              # osso
        return a

    def _janela_lateral(self, n_ativos):
        """
        Janela w(u) da Eq. 4, soma unitaria. A Eq. 4 integra de x - l a x + l com l = n_a * pitch
        (largura total 2 n_a pitch, janela_literal=True); a Fig. 10 informa a "Aper. Width" em linhas
        de varredura, o que sugere largura total n_a * pitch (padrao). A opcao literal borra
        bastante o speckle quando a PSF lateral ja e realista.
        """
        largura = n_ativos * self.pitch_mm * (2.0 if self.janela_literal else 1.0)
        n = max(int(round(largura / self.dx_fino)) | 1, 3)
        u = np.linspace(-1.0, 1.0, n)
        if self.janela == "hann":
            w = 0.5 * (1.0 + np.cos(np.pi * u))
        elif self.janela == "triangular":
            w = 1.0 - np.abs(u)
        else:
            w = np.ones(n)
        w = w[1:-1] if self.janela != "quadrada" else w     # extremos nulos da Hann/triangular
        return w / w.sum()

    def simular_fatia(self, hu_slice, corpo_slice, roi_slice, rn_slice, m_slice, coords, spacing):
        """
        Devolve (I_us, I_r, I_s) em intensidade na grade fina da regiao da ROI.
        """
        dx, dy, _ = spacing
        hu_slice = np.where(corpo_slice, hu_slice, 0.0).astype(np.float32)
        hu = ndimage.map_coordinates(hu_slice, coords, order=1, mode="nearest")
        if self.suavizacao_ct_z_mm > 0:
            # suavizacao normalizada so dentro do corpo: a interface agua -> pele continua nitida
            sg = (self.suavizacao_ct_z_mm / dy, self.suavizacao_ct_z_mm / dx)
            peso = ndimage.gaussian_filter(corpo_slice.astype(np.float32), sg)
            hu_z_slice = ndimage.gaussian_filter(hu_slice * corpo_slice, sg) / np.maximum(peso, 1e-3)
            hu_z_slice = np.where(corpo_slice, hu_z_slice, 0.0).astype(np.float32)
            hu_z = ndimage.map_coordinates(hu_z_slice, coords, order=1, mode="nearest")
        else:
            hu_z = hu
        corpo_s = ndimage.gaussian_filter(corpo_slice.astype(np.float32), 1.0)
        corpo = ndimage.map_coordinates(corpo_s, coords, order=1, mode="constant") > 0.5
        w = ndimage.map_coordinates(roi_slice.astype(np.float32), coords, order=1, mode="constant") * corpo
        rn = ndimage.map_coordinates(rn_slice, coords, order=1, mode="nearest")
        m = ndimage.map_coordinates(m_slice, coords, order=1, mode="constant") * w

        # Fora do corpo / ROI -> agua (0 HU)
        hu = hu * w
        hu_z = hu_z * w

        # 1) Z: linear no HU para tecido mole; ar e osso segmentados por limiar [KSN sec. 2.1]
        Z = Z_AGUA * (hu_z + 1000.0) / 1000.0
        Z = np.where(hu_z < self.limiar_ar_hu, Z_AR, Z)
        Z = np.where(hu_z > self.limiar_osso_hu, self.z_osso, Z)
        Z = np.maximum(Z, Z_AR)

        # 2) alpha_R [Eq. 1] na escala de um voxel da TC + maximo local ao longo do feixe
        h = max(int(round(dy / (2.0 * self.dz_fino))), 1)
        Zp = np.pad(Z, ((0, h), (0, 0)), mode="edge")[h:]
        Zm = np.pad(Z, ((h, 0), (0, 0)), mode="edge")[:-h]
        Rd = ((Zp - Zm) / (Zp + Zm)) ** 2
        pico = np.zeros_like(Rd, dtype=bool)
        pico[1:-1] = (Rd[1:-1] >= Rd[:-2]) & (Rd[1:-1] > Rd[2:]) & (Rd[1:-1] > 1e-6)
        aR = np.where(pico, Rd, 0.0).astype(np.float32)

        # 3) I_i(x) = I_i(x - d) (1 - alpha_R(x - d))  [Eq. 5], I_0 = 1
        Ii = np.cumprod(1.0 - aR, axis=0)
        Ii = np.vstack([np.ones((1, Ii.shape[1]), Ii.dtype), Ii[:-1]])
        dz_cm = self.dz_fino / 10.0
        if self.absorcao != "nenhuma":                              # extensao (fora do artigo)
            # So com a Eq. 5 a sombra de uma costela fica em ~ -11 dB (R ~ 0,46 na entrada e na
            # saida). Na realidade a sombra vem sobretudo da ABSORCAO no osso (~10-20 dB/cm/MHz).
            # "osso": absorcao so no osso e no ar segmentados (tecido mole sem absorcao, como no
            # artigo, que equivale a TGC perfeito); "todas": absorcao em todos os tecidos.
            alpha = self.alpha_por_hu(hu) * w + ALPHA_AGUA * (1.0 - w)
            if self.absorcao == "osso":
                alpha = np.where((hu > self.limiar_osso_hu) | (hu < self.limiar_ar_hu), alpha, 0.0)
            Ii = Ii * 10.0 ** (-np.cumsum(alpha * self.f0_mhz * dz_cm, axis=0) / 10.0)
        aT_acum = Ii                                                # transmissao acumulada (Figs. 5a, 6a)

        # 4-5) I_r ~ alpha_R I_i^2/I_0 |r.n|, integrado na abertura ativa  [Eq. 3-4]
        Ir = aR * Ii ** 2 * rn
        Ir = ndimage.convolve1d(Ir, self._janela_lateral(self.na_reflexao), axis=1, mode="constant")

        # 6) I_s: espalhadores -> PSF -> envelope^2, "alongado" pela mesma janela [sec. 2.3]
        rf = self._espalhamento_rf(m)
        Is = np.abs(hilbert(rf, axis=0)) ** 2 / self._normalizacao_is()
        if self.na_espalhamento > 0:
            Is = ndimage.convolve1d(Is, self._janela_lateral(self.na_espalhamento), axis=1, mode="nearest")

        # 7) I_us = (G_s1 * I_r + alpha G_s2 * alpha_T) I_s  [Eq. 6]
        s1 = (self.sigma_reflexao_mm / self.dz_fino, self.sigma_reflexao_mm / self.dx_fino)
        s2 = (self.sigma_transmissao_mm / self.dz_fino, self.sigma_transmissao_mm / self.dx_fino)
        G1_Ir = ndimage.gaussian_filter(Ir, s1, mode="nearest") if self.sigma_reflexao_mm > 0 else Ir
        if self.normalizar_pico_reflexao and self.sigma_reflexao_mm > 0:
            G1_Ir = G1_Ir * np.sqrt(2.0 * np.pi) * s1[0]           # pico de uma interface preservado
        G2_aT = ndimage.gaussian_filter(aT_acum, s2, mode="nearest") if self.sigma_transmissao_mm > 0 else aT_acum
        Ius = (G1_Ir + self.alpha_mistura * G2_aT) * Is

        # TGC opcional (o artigo nao usa; so faz sentido com absorcao=True)
        if self.tgc_db_cm_mhz:
            prof_cm = np.arange(Ius.shape[0])[:, None] * dz_cm
            Ius = Ius * 10.0 ** (2.0 * self.tgc_db_cm_mhz * self.f0_mhz * prof_cm / 10.0)
        return Ius.astype(np.float32), G1_Ir.astype(np.float32), Is.astype(np.float32)

    def comprimir_log(self, intensidade, percentil: float = 99.5):
        """
        [KSN Eq. 7]: 10 log10(I/max) limitado a beta = faixa_dinamica_db; devolvido em [-beta, 0]
        (a Eq. 7 soma beta e da [0, beta]). referencia_log="percentil" usa o percentil em vez do max.
        """
        if self.referencia_log == "max":
            ref = float(intensidade.max())
        else:
            ref = np.percentile(intensidade[intensidade > 0], percentil)
        db = 10.0 * np.log10(np.maximum(intensidade / ref, 1e-12))
        return np.clip(db, -self.faixa_dinamica_db, 0.0).astype(np.float32)


# --- ROI FREEHAND ---

class SelecionadorROIFreehand:
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


def gerar_mascara_3d_a_partir_de_roi_freehand(volume_shape, mascara_2d):
    mascara_3d = np.broadcast_to(mascara_2d.astype(np.float32), volume_shape).copy()
    return ndimage.gaussian_filter(mascara_3d, sigma=(0, 2.0, 2.0))


# --- VISUALIZACAO ---

def visualizar_fatiador_interativo(volume_3d, extent, faixa_db):
    fig, ax = plt.subplots(figsize=(7, 8))
    plt.subplots_adjust(bottom=0.2)
    k0 = volume_3d.shape[0] // 2
    img = ax.imshow(volume_3d[k0], cmap='gray', vmin=-faixa_db, vmax=0, extent=extent)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
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


def visualizar_volume_3d_pyvista(volume_3d, spacing_zyx, faixa_db):
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
    plotter.show(title="Volume 3D Ultrassonico (modelo de Kutter)")


# --- MAIN ---
if __name__ == "__main__":
    caminho_serie_dicom = r"C:\Users\TOMMUS\Downloads\QIN-BREAST_2015-09-04\qin_breast\QIN-BREAST-01-0002\1992-01-24-BREAST PRONE-12000\2-CTAC-14691"

    sim = SimuladorKutter(f0_mhz=5.0)

    if not os.path.exists(caminho_serie_dicom):
        raise SystemExit(f"Erro: O caminho especificado nao existe:\n{caminho_serie_dicom}")

    print("1. Carregando e reorientando o volume DICOM (LPS)...")
    volume_hu, spacing = sim.carregar_serie_dicom(caminho_serie_dicom)
    num_fatias = volume_hu.shape[0]
    print(f"   -> Volume: {volume_hu.shape} | voxel (dx, dy, dz) = {spacing} mm")

    print("2. Segmentando o corpo (ar externo -> agua)...")
    corpo = sim.mascara_corpo(volume_hu)

    print("\n==============================================")
    print(" ESCOLHA O MODO DE PROCESSAMENTO:")
    print(" [1] Desenhar ROI Freehand (Desenho com o mouse)")
    print(" [2] Usar Imagem Inteira (Sem ROI)")
    print("==============================================")
    opcao = input("Digite 1 ou 2 e pressione Enter: ").strip()

    mascara_2d = None
    if opcao == "1":
        seletor = SelecionadorROIFreehand(volume_hu[num_fatias // 2])
        if np.any(seletor.mascara_2d):
            mascara_2d = seletor.mascara_2d
            print("   -> ROI Freehand capturada.")
        else:
            print("   -> Nenhuma ROI desenhada. Processando imagem inteira.")
    if mascara_2d is not None:
        roi_3d = gerar_mascara_3d_a_partir_de_roi_freehand(volume_hu.shape, mascara_2d)
    else:
        roi_3d = np.ones_like(volume_hu, dtype=np.float32)

    print("3. Normais (Deriche 3D) e mapa de espalhamento...")
    t0 = time.time()
    rn, m_esp, gmag = sim.preprocessar_volume(volume_hu, corpo, spacing,
                                              roi_3d if mascara_2d is not None else None)
    ip = sim.info_preproc
    print(f"   -> Estiramento: {ip['hu_lo']:.0f} a {ip['hu_hi']:.0f} HU | "
          f"bordas: {100 * ip['frac_bordas']:.1f}% dos voxels | {time.time() - t0:.1f} s")

    print("4. Regiao simulada (ROI)...")
    x0, x1, topo, fundo = sim.regiao_da_roi(corpo, spacing, mascara_2d)
    coords, prof, xs = sim.coordenadas_regiao(spacing, x0, x1, topo, fundo)
    nzf, nxf = coords.shape[1:]
    print(f"   -> x = {x0:.1f} a {x1:.1f} mm, y = {topo:.1f} a {fundo:.1f} mm | grade {nzf} x {nxf}")
    print(f"   -> Janela {sim.janela}: {sim.na_reflexao} elementos (reflexao), {sim.na_espalhamento} (espalhamento), "
          f"pitch {sim.pitch_mm} mm | alpha = {sim.alpha_mistura} | sigma_R = {sim.sigma_reflexao_mm} mm, "
          f"sigma_T = {sim.sigma_transmissao_mm} mm")

    print("5. Simulando fatia por fatia...")
    nz_out = len(range(0, nzf, 2))
    vol_us = np.empty((num_fatias, nz_out, nxf), np.float32)
    k_meio = num_fatias // 2
    for k in range(num_fatias):
        Ius, Ir, Is = sim.simular_fatia(volume_hu[k], corpo[k], roi_3d[k], rn[k], m_esp[k], coords, spacing)
        vol_us[k] = Ius[::2]
        if k == k_meio:
            Ir_meio, Is_meio = Ir[::2], Is[::2]
        print(f"\r   -> fatia {k + 1}/{num_fatias}", end="", flush=True)
    print()
    vol_db = sim.comprimir_log(vol_us)
    dz_out = sim.dz_fino * 2

    # --- FIGURA: TC, I_r, I_s, I_us ---
    dx, dy, _ = spacing
    ny, nx = volume_hu.shape[1:]
    ext_ct = [0, nx * dx, ny * dy, 0]
    ext_us = [x0, x1, topo + prof[-1], topo]
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    ax[0].imshow(volume_hu[k_meio], cmap='bone', vmin=-150, vmax=200, extent=ext_ct)
    if mascara_2d is not None:
        ax[0].contour(mascara_2d, levels=[0.5], colors='red', linewidths=1, extent=[0, nx * dx, 0, ny * dy])
    ax[0].set_title("1. TC (LPS) + ROI", fontweight='bold')
    db = lambda I: 10 * np.log10(np.maximum(I / np.percentile(I[I > 0], 99.5), 1e-12))
    ax[1].imshow(db(Ir_meio + 1e-12), cmap='gray', vmin=-sim.faixa_dinamica_db, vmax=0, extent=ext_us)
    ax[1].set_title("2. Reflexao G1*I_r", fontweight='bold')
    ax[2].imshow(db(Is_meio + 1e-12), cmap='gray', vmin=-sim.faixa_dinamica_db, vmax=0, extent=ext_us)
    ax[2].set_title("3. Espalhamento I_s", fontweight='bold')
    ax[3].imshow(vol_db[k_meio], cmap='gray', vmin=-sim.faixa_dinamica_db, vmax=0, extent=ext_us)
    ax[3].set_title("4. US simulado [Eq. 6]", fontweight='bold')
    for a in ax:
        a.set_xlabel("x (mm)")
    ax[0].set_ylabel("y (mm)")
    plt.tight_layout()
    plt.show()

    visualizar_fatiador_interativo(vol_db, ext_us, sim.faixa_dinamica_db)
    visualizar_volume_3d_pyvista(vol_db, (spacing[2], dz_out, sim.dx_fino), sim.faixa_dinamica_db)
