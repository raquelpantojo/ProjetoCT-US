"""
Referencia do banco de dados:
Li, X., Abramson, R. G., Arlinghaus, L. R., Chakravarthy, A. B., Abramson, V. G., Sanders, M., & Yankeelov, T. E. (2016).
Data From QIN-BREAST (Version 2) [Data set].
The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.21JUEBH0

Instalar:
pip install SimpleITK scipy numpy matplotlib pyvista

Visao geral do modelo (versao revisada)
---------------------------------------
1. Le a serie CT e reorienta para LPS (linha 0 = anterior). Como a aquisicao e em
   decubito ventral (HFP), as mamas ficam no topo da imagem e o transdutor e
   posicionado ali, com o feixe propagando no sentido anterior -> posterior.
2. Ar externo ao corpo e substituido por agua (meio de acoplamento).
3. Para cada fatia, a regiao simulada e a ROI desenhada no inicio (ou o corpo inteiro,
   sem ROI): usa-se o retangulo envolvente da ROI, com o topo numa camada de agua logo
   acima da pele, e tudo fora da ROI vira agua. Essa regiao e reamostrada para uma
   grade fina (lambda/8 axial, lambda/4 lateral). Isso e
   obrigatorio: o voxel da TC (1,37 mm) e ~9x maior que o comprimento de onda
   (0,31 mm a 5 MHz), entao a portadora e o speckle nao podem ser representados
   na grade original.
4. Classes de tecido -> rho, c, alpha (valores de referencia, ajustaveis).
5. Refletividade especular R = (Z2 - Z1)/(Z2 + Z1) + espalhadores sub-resolucao.
6. Atenuacao ida-e-volta (usa alpha_0) e perda de transmissao prod(1 - R^2).
7. Convolucao com PSF em unidades fisicas (f0, banda fracionaria, F-number),
   envelope por Hilbert, TGC, normalizacao GLOBAL do volume e compressao log.
"""

import os
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


class PipelineTCparaUltrassom:
    """
    Conversao de volumes DICOM de Tomografia em imagens B-mode de Ultrassom.
    """

    def __init__(self, f0_mhz: float = 5.0, c0_m_s: float = 1540.0,
                 banda_fracionaria: float = 0.6, f_number: float = 2.0,
                 faixa_dinamica_db: float = 50.0, tgc_db_cm_mhz: float = 0.5,
                 semente: int = 0):
        self.f0_mhz = f0_mhz
        self.f0 = f0_mhz * 1e6
        self.c0 = c0_m_s
        self.lambda_mm = self.c0 / self.f0 * 1e3          # 0,308 mm a 5 MHz
        self.banda = banda_fracionaria                    # FWHM / f0 (-6 dB)
        self.f_number = f_number
        self.faixa_dinamica_db = faixa_dinamica_db
        self.tgc_db_cm_mhz = tgc_db_cm_mhz                # compensacao media (ida e volta)
        self.rng = np.random.default_rng(semente)          # reprodutibilidade

        # Grade fina de simulacao (mm)
        self.dz_fino = self.lambda_mm / 8.0                # 4 amostras por periodo da portadora (lambda/2)
        self.dx_fino = self.lambda_mm / 4.0

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
        Separa o corpo da mesa/suporte e do ar externo.
        O limiar deve ser IGUAL ao da classe "ar" em calcular_propriedades_acusticas:
        se for menor, voxels de pele com volume parcial (ex.: -450 HU) entram no corpo
        classificados como ar e geram sombra total na coluna.
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
    # HU -> propriedades acusticas
    # ------------------------------------------------------------------ #
    @staticmethod
    def calcular_propriedades_acusticas(hu: np.ndarray):
        """
        Mapeamento por classes de tecido. A descontinuidade entre classes e intencional:
        ela gera a interface especular na isolinha de HU (sub-voxel, pois o HU e
        interpolado antes da classificacao). Dentro de cada classe, variacao linear suave.

        Valores de referencia (ajuste/cite conforme a sua fonte):
        - Duck, F. A. (1990). Physical Properties of Tissue.
        - IT'IS Foundation Tissue Properties Database v4.1.
        - Mast, T. D. (2000). ARLO 1, 37-42 (relacoes empiricas em tecidos moles).
        """
        rho = np.empty_like(hu, dtype=np.float32)
        c = np.empty_like(hu, dtype=np.float32)
        alpha_0 = np.empty_like(hu, dtype=np.float32)    # dB/cm/MHz

        m_ar = hu < -400
        m_gord = (hu >= -400) & (hu < -30)
        m_gland = (hu >= -30) & (hu <= 150)
        m_osso = hu > 150

        # 1. Ar (pulmao): Z ~ 400 Rayl -> R ~ -1 -> sombra acustica total
        rho[m_ar], c[m_ar], alpha_0[m_ar] = 1.2, 343.0, 12.0

        # 2. Gordura (~ -100 HU): ~920-950 kg/m3, ~1440-1470 m/s
        h = np.clip(hu[m_gord], -150.0, -30.0)
        rho[m_gord] = 925.0 + 0.25 * (h + 100.0)
        c[m_gord] = 1450.0 + 0.30 * (h + 100.0)
        alpha_0[m_gord] = 0.60

        # 3. Fibroglandular / musculo (~ +40 HU): ~1040 kg/m3, ~1540 m/s
        h = hu[m_gland]
        rho[m_gland] = 1040.0 + 0.25 * (h - 40.0)
        c[m_gland] = 1540.0 + 0.40 * (h - 40.0)
        alpha_0[m_gland] = 0.75

        # 4. Osso (costelas/esterno). Obs.: microcalcificacoes NAO sao resolvidas
        #    em CTAC de baixa dose (1,37 mm x 3,75 mm); >150 HU aqui e praticamente osso.
        h = np.clip(hu[m_osso], 150.0, 1500.0)
        rho[m_osso] = 1100.0 + 0.60 * (h - 150.0)          # ~1900 kg/m3 em osso cortical
        c[m_osso] = 1700.0 + 1.35 * (h - 150.0)            # ~3500 m/s em osso cortical
        alpha_0[m_osso] = 10.0

        Z = rho * c
        return rho, c, Z, alpha_0

    def sigma_espalhadores(self, hu: np.ndarray, corpo: np.ndarray) -> np.ndarray:
        """
        Desvio padrao relativo da impedancia dos espalhadores sub-resolucao.
        Mesmos limiares da classificacao (antes eram -30/150 vs -20/120).
        Valores calibrados para que o speckle fique ~10-20 dB abaixo das interfaces.
        """
        s = np.zeros_like(hu, dtype=np.float32)
        s[(hu >= -400) & (hu < -30)] = 0.010      # gordura: hipoecoica
        s[(hu >= -30) & (hu <= 150)] = 0.025      # fibroglandular: hiperecoica
        s[hu > 150] = 0.010
        s[~corpo] = 0.0                           # agua: anecoica
        return s

    # ------------------------------------------------------------------ #
    # PSF em unidades fisicas
    # ------------------------------------------------------------------ #
    def gerar_psf_transdutor(self):
        """
        PSF 2D (axial x lateral) amostrada na grade fina.
        Axial: pulso gaussiano de banda fracionaria B, ida e volta -> portadora cos(4*pi*z/lambda).
        Lateral: gaussiana com FWHM ~ lambda * F#.
        """
        sigma_t = 2.3548 / (2 * np.pi * self.banda * self.f0)     # s
        sigma_z = self.c0 * sigma_t / 2.0 * 1e3                   # mm (profundidade)
        sigma_x = self.lambda_mm * self.f_number / 2.3548         # mm

        z = np.arange(-4 * sigma_z, 4 * sigma_z + self.dz_fino, self.dz_fino)
        x = np.arange(-4 * sigma_x, 4 * sigma_x + self.dx_fino, self.dx_fino)
        zz, xx = np.meshgrid(z, x, indexing="ij")

        envelope = np.exp(-0.5 * ((zz / sigma_z) ** 2 + (xx / sigma_x) ** 2))
        portadora = np.cos(4 * np.pi * zz / self.lambda_mm)
        psf = envelope * portadora
        self.psf_info = dict(sigma_z_mm=sigma_z, sigma_x_mm=sigma_x,
                             fwhm_axial_mm=2.3548 * sigma_z, fwhm_lateral_mm=2.3548 * sigma_x,
                             shape=psf.shape)
        return psf / np.sum(np.abs(psf))

    # ------------------------------------------------------------------ #
    # Regiao simulada = ROI
    # ------------------------------------------------------------------ #
    @staticmethod
    def regiao_da_roi(corpo: np.ndarray, spacing, mascara_2d=None,
                      folga_mm: float = 3.0, margem_mm: float = 2.0):
        """
        Retangulo envolvente (em mm, coordenadas da TC) da regiao a simular.
        - Com ROI: colunas e linhas ocupadas pela ROI; o topo e posto 'folga_mm' acima da
          primeira linha em que a ROI encontra o corpo (pele), numa camada de agua, para
          que o feixe (anterior -> posterior) comece no meio de acoplamento.
        - Sem ROI: o mesmo, usando o corpo inteiro (todas as fatias).
        Devolve (x0_mm, x1_mm, topo_mm, fundo_mm).
        """
        dx, dy, _ = spacing
        ny, nx = corpo.shape[1:]
        proj_corpo = corpo.any(axis=0)                            # (Y, X)
        regiao = proj_corpo if mascara_2d is None else mascara_2d.astype(bool)
        tecido = regiao & proj_corpo
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

    def coordenadas_regiao(self, spacing, x0_mm, x1_mm, topo_mm, fundo_mm):
        """Coordenadas (em indices da TC) da grade fina que cobre a regiao da ROI."""
        dx, dy, _ = spacing
        nzf = int(round((fundo_mm - topo_mm) / self.dz_fino))
        nxf = int(round((x1_mm - x0_mm) / self.dx_fino))
        prof = np.arange(nzf) * self.dz_fino                      # profundidade a partir do topo
        xs = x0_mm + (np.arange(nxf) + 0.5) * self.dx_fino        # x absoluto (mm) na TC
        # centro do voxel i esta em (i + 0.5) * dx
        lin = (topo_mm + prof) / dy - 0.5
        col = xs / dx - 0.5
        LL, CC = np.meshgrid(lin, col, indexing="ij")
        return np.stack([LL, CC]), prof, xs

    # ------------------------------------------------------------------ #
    # Simulacao de uma fatia
    # ------------------------------------------------------------------ #
    def simular_envelope_2d(self, hu_slice, corpo_slice, roi_slice, coords, psf):
        """
        Devolve o envelope (linear, sem normalizar) de uma fatia na regiao da ROI.
        """
        # O ar EXTERNO vira agua antes da interpolacao. Sem isso, perto da pele o HU
        # interpolado (entre -1000 e tecido) cai na classe "ar" -> R ~ -1 -> sombra total
        # em colunas aleatorias. O ar INTERNO (pulmao) e mantido.
        hu_slice = np.where(corpo_slice, hu_slice, 0.0).astype(np.float32)

        # HU interpolado (bilinear) -> classes com bordas sub-voxel suaves
        hu = ndimage.map_coordinates(hu_slice, coords, order=1, mode="nearest")
        # Mascara levemente suavizada antes de interpolar: contorno da pele sem "escada" de 1 voxel
        corpo_suave = ndimage.gaussian_filter(corpo_slice.astype(np.float32), sigma=1.0)
        corpo = ndimage.map_coordinates(corpo_suave, coords, order=1, mode="constant") > 0.5
        peso_roi = ndimage.map_coordinates(roi_slice.astype(np.float32), coords, order=1, mode="constant", cval=0.0)

        _, _, Z, alpha_0 = self.calcular_propriedades_acusticas(hu)

        # Fora do corpo -> agua; fora da ROI -> agua (transicao suave da apodizacao)
        Z_agua = RHO_AGUA * C_AGUA
        w = peso_roi * corpo
        Z = Z * w + Z_agua * (1.0 - w)
        alpha_0 = alpha_0 * w + ALPHA_AGUA * (1.0 - w)

        # Espalhadores sub-resolucao (a grade fina e menor que a PSF -> speckle real)
        sigma = self.sigma_espalhadores(hu, corpo) * w
        Z_micro = Z * (1.0 + sigma * self.rng.standard_normal(Z.shape).astype(np.float32))

        # Coeficiente de reflexao entre amostras consecutivas no eixo axial
        R_macro = np.zeros_like(Z)
        R_macro[1:] = (Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1])
        R_total = np.zeros_like(Z)
        R_total[1:] = (Z_micro[1:] - Z_micro[:-1]) / (Z_micro[1:] + Z_micro[:-1])

        # Perda de transmissao nas interfaces (ida e volta): prod(1 - R^2)^2
        T = np.cumprod(1.0 - R_macro ** 2, axis=0)
        T = np.vstack([np.ones((1, T.shape[1]), T.dtype), T[:-1]]) ** 2

        # Atenuacao ida e volta: 2 * alpha0[dB/cm/MHz] * f0[MHz] * z[cm]
        dz_cm = self.dz_fino / 10.0
        atten_db = 2.0 * np.cumsum(alpha_0 * self.f0_mhz * dz_cm, axis=0)
        A = 10.0 ** (-atten_db / 20.0)

        refletividade = R_total * T * A
        sinal_rf = fftconvolve(refletividade, psf, mode="same")
        envelope = np.abs(hilbert(sinal_rf, axis=0))

        # TGC: compensa uma atenuacao media de referencia (dB/cm/MHz, ida e volta)
        prof_cm = np.arange(envelope.shape[0])[:, None] * dz_cm
        envelope *= 10.0 ** (2.0 * self.tgc_db_cm_mhz * self.f0_mhz * prof_cm / 20.0)
        return envelope

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
    Obs.: a mesma ROI vale para todas as fatias; a anatomia muda ao longo de Z.
    """
    mascara_3d = np.broadcast_to(mascara_2d.astype(np.float32), volume_shape).copy()
    return ndimage.gaussian_filter(mascara_3d, sigma=(0, 2.0, 2.0))


# --- FUNCOES DE VISUALIZACAO INTERATIVA ---

def visualizar_fatiador_interativo(volume_3d: np.ndarray, extent, faixa_db: float):
    fig, ax = plt.subplots(figsize=(7, 8))
    plt.subplots_adjust(bottom=0.2)
    k0 = volume_3d.shape[0] // 2
    img = ax.imshow(volume_3d[k0], cmap='gray', vmin=-faixa_db, vmax=0, extent=extent)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
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

    print("Inicializando a Conversao de TC para ultrassom...")
    pipeline = PipelineTCparaUltrassom(f0_mhz=5)

    if not os.path.exists(caminho_serie_dicom):
        raise SystemExit(f"Erro: O caminho especificado nao existe:\n{caminho_serie_dicom}")

    print("1. Carregando e reorientando o volume DICOM (LPS)...")
    volume_hu, spacing = pipeline.carregar_serie_dicom(caminho_serie_dicom)
    num_fatias = volume_hu.shape[0]
    print(f"   -> Volume: {volume_hu.shape} (Fatias x Linhas x Colunas)")
    print(f"   -> Voxel (dx, dy, dz): {spacing} mm | lambda = {pipeline.lambda_mm:.3f} mm")

    print("2. Segmentando o corpo (ar externo -> agua)...")
    corpo = pipeline.mascara_corpo(volume_hu)

    # --- SELECAO DO MODO DE RECORTE ---
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
    else:
        mascara_roi_3d = np.ones_like(volume_hu, dtype=np.float32)

    print("3. Definindo a regiao simulada (ROI) e gerando a PSF...")
    x0_mm, x1_mm, topo_mm, fundo_mm = pipeline.regiao_da_roi(corpo, spacing, mascara_2d)
    coords, prof, xs = pipeline.coordenadas_regiao(spacing, x0_mm, x1_mm, topo_mm, fundo_mm)
    psf = pipeline.gerar_psf_transdutor()
    nzf, nxf = coords.shape[1:]
    mem_gb = num_fatias * (nzf // 2 + nzf % 2) * nxf * 4 / 1e9
    print(f"   -> Regiao: x = {x0_mm:.1f} a {x1_mm:.1f} mm, y = {topo_mm:.1f} a {fundo_mm:.1f} mm "
          f"({x1_mm - x0_mm:.1f} x {fundo_mm - topo_mm:.1f} mm)")
    print(f"   -> Grade fina por fatia: {nzf} x {nxf} | volume final ~{mem_gb:.2f} GB")
    print(f"   -> PSF: FWHM axial {pipeline.psf_info['fwhm_axial_mm']:.2f} mm, "
          f"lateral {pipeline.psf_info['fwhm_lateral_mm']:.2f} mm, {psf.shape} amostras")

    print("4. Simulando o volume fatia por fatia...")
    # Decima o eixo axial por 2 ja em cada fatia (o envelope tem banda bem menor que a
    # portadora); assim o volume completo na grade fina nunca fica todo na memoria.
    envelopes = np.empty((num_fatias, len(range(0, nzf, 2)), nxf), dtype=np.float32)
    for k in range(num_fatias):
        envelopes[k] = pipeline.simular_envelope_2d(volume_hu[k], corpo[k], mascara_roi_3d[k], coords, psf)[::2]
        print(f"\r   -> fatia {k + 1}/{num_fatias}", end="", flush=True)
    print()

    volume_ultrassom_3d = pipeline.comprimir_log(envelopes)
    dz_out = pipeline.dz_fino * 2
    print(f"   -> Volume concluido: {volume_ultrassom_3d.shape}, "
          f"voxel (fatia, prof, lat) = ({spacing[2]:.2f}, {dz_out:.3f}, {pipeline.dx_fino:.3f}) mm")

    # --- FIGURA COMPARATIVA ---
    k = num_fatias // 2
    # Mesmas coordenadas da TC (mm) nos tres paineis
    extent_us = [x0_mm, x1_mm, topo_mm + prof[-1], topo_mm]
    dx, dy, _ = spacing
    ny, nx = volume_hu.shape[1:]
    extent_ct = [0, nx * dx, ny * dy, 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    im0 = axes[0].imshow(volume_hu[k], cmap='bone', vmin=-150, vmax=200, extent=extent_ct)
    if mascara_2d is not None:
        axes[0].contour(mascara_2d, levels=[0.5], colors='red', linewidths=1, extent=[0, nx * dx, 0, ny * dy])
        axes[0].set_title("1. TC (LPS, anterior no topo) + ROI", fontsize=11, fontweight='bold')
    else:
        axes[0].set_title("1. TC (LPS, anterior no topo)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("x (mm)"); axes[0].set_ylabel("y (mm)")
    fig.colorbar(im0, ax=axes[0], label="HU")

    hu_win = ndimage.map_coordinates(volume_hu[k], coords, order=1, mode="nearest")
    hu_win = np.where(ndimage.map_coordinates(mascara_roi_3d[k], coords, order=1) > 0.5, hu_win, np.nan)
    im1 = axes[1].imshow(hu_win, cmap='bone', vmin=-150, vmax=200, extent=extent_us, aspect='equal')
    axes[1].set_title("2. TC reamostrada na ROI", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("x (mm)"); axes[1].set_ylabel("y (mm)")
    fig.colorbar(im1, ax=axes[1], label="HU")

    im2 = axes[2].imshow(volume_ultrassom_3d[k], cmap='gray', vmin=-pipeline.faixa_dinamica_db, vmax=0,
                         extent=extent_us, aspect='equal')
    axes[2].set_title("3. Ultrassom simulado (B-mode)", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("x (mm)")
    fig.colorbar(im2, ax=axes[2], label="dB")
    plt.tight_layout()
    plt.show()

    visualizar_fatiador_interativo(volume_ultrassom_3d, extent_us, pipeline.faixa_dinamica_db)
    visualizar_volume_3d_pyvista(volume_ultrassom_3d, (spacing[2], dz_out, pipeline.dx_fino),
                                 pipeline.faixa_dinamica_db)
