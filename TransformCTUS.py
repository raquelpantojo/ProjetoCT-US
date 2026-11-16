"""

Codigo escrito por Raquel Pantojo de Souza Bachour
 
Referencia do banco de dados:
Li, X., Abramson, R. G., Arlinghaus, L. R., Chakravarthy, A. B., Abramson, V. G., Sanders, M., & Yankeelov, T. E. (2016). 
Data From QIN-BREAST (Version 2) [Data set]. 
The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.21JUEBH0


Instalar:
pip install SimpleITK scipy numpy matplotlib pydicom pyvista


"""

import os
import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, LassoSelector
from matplotlib.path import Path
import SimpleITK as sitk
import pyvista as pv


class PipelineTCparaUltrassom:
    """
    Conversão de volumes DICOM de Tomografia 
    em imagens B-mode de Ultrassom utilizando modelos físicos.
    """
    def __init__(self, f0_mhz: float = 5, c0_m_s: float = 1540.0):
        self.f0 = f0_mhz * 1e6              # Frequência central (5 MHz)
        self.c0 = c0_m_s                    # Velocidade de referência (m/s)
        self.lambda_m = self.c0 / self.f0    # Comprimento de onda acústica (m)

    def carregar_serie_dicom(self, caminho_pasta: str):
        """
        Lê a série DICOM completa do diretório e extrai a matriz HU 3D com a calibração física.
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(caminho_pasta)
        
        if not dicom_names:
            raise FileNotFoundError(f"Nenhum arquivo DICOM foi encontrado em: {caminho_pasta}")
            
        reader.SetFileNames(dicom_names)
        imagem_sitk = reader.Execute()
        
        # Converte para matriz NumPy em float32 (Z, Y, X)
        volume_hu = sitk.GetArrayFromImage(imagem_sitk).astype(np.float32)
        # Resolução dos voxels em mm (dx, dy, dz)
        spacing = imagem_sitk.GetSpacing()
        
        return volume_hu, spacing

    def calcular_propriedades_acusticasv2(self, volume_hu: np.ndarray):
        """
        Mapeia a matriz HU em Densidade (rho), Velocidade (c), Atenuação (alpha) e Impedância (Z).
        """
        # 1. Densidade Mássica rho (kg/m^3) via Curva Bilinear de Schneider
        rho = np.zeros_like(volume_hu, dtype=np.float32)
        rho[volume_hu < 0] = 1000.0 + 0.93 * volume_hu[volume_hu < 0]
        rho[volume_hu >= 0] = 1000.0 + 0.45 * volume_hu[volume_hu >= 0]
        rho = np.clip(rho, 1.2, 2200.0)

        # 2. Velocidade da Onda c (m/s)
        c = 1540.0 + 0.58 * (rho - 1040.0)
        c[volume_hu < -50] = 1450.0 + 0.40 * (rho[volume_hu < -50] - 920.0)
        c = np.clip(c, 1400.0, 3200.0)

        # 3. Impedância Acústica Z (Pa.s/m)
        Z = rho * c

        # 4. Coeficiente de Atenuação Acústica alpha_0 (dB/cm/MHz)
        alpha_0 = 0.53 * (rho / 1000.0)**1.8

        return rho, c, Z, alpha_0

    def calcular_propriedades_acusticas(self, volume_hu: np.ndarray):
        """
        Mapeamento Físico-Anatômico Trilinear para Tecido Mamário.
        Referências: 
        - Duck, F. A. (1990). Physical Properties of Tissue.
        - Culjat et al. (2010). A Review of Tissue Substitutes for Ultrasound Imaging Phantoms.
        """
        rho = np.zeros_like(volume_hu, dtype=np.float32)
        c = np.zeros_like(volume_hu, dtype=np.float32)
        alpha_0 = np.zeros_like(volume_hu, dtype=np.float32)

        # Máscaras de classificação baseadas em física tecidual
        mask_ar = volume_hu < -300
        mask_gordura = (volume_hu >= -300) & (volume_hu < -20)
        mask_glandular = (volume_hu >= -20) & (volume_hu <= 120)
        mask_calc = volume_hu > 120

        # 1. Ar / Meio Externo
        rho[mask_ar] = 1.2
        c[mask_ar] = 343.0
        alpha_0[mask_ar] = 12.0

        # 2. Tecido Adiposo (Gordura Mamária: ~920 kg/m^3, ~1450 m/s)
        rho[mask_gordura] = 920.0 + 0.8 * (volume_hu[mask_gordura] + 120.0)
        c[mask_gordura] = 1450.0 + 0.3 * (volume_hu[mask_gordura] + 120.0)
        alpha_0[mask_gordura] = 0.63

        # 3. Tecido Fibroglandular / Parênquima (~1040 kg/m^3, ~1550 m/s)
        rho[mask_glandular] = 1030.0 + 0.4 * volume_hu[mask_glandular]
        c[mask_glandular] = 1540.0 + 0.5 * volume_hu[mask_glandular]
        alpha_0[mask_glandular] = 0.85

        # 4. Microcalcificações / Tecido Denso
        rho[mask_calc] = 1400.0 + 0.6 * (volume_hu[mask_calc] - 120.0)
        c[mask_calc] = 2200.0 + 0.8 * (volume_hu[mask_calc] - 120.0)
        alpha_0[mask_calc] = 3.00

        # Limites físicos de segurança (Clipping)
        rho = np.clip(rho, 1.2, 2200.0)
        c = np.clip(c, 343.0, 3000.0)

        # 5. Impedância Acústica Z = rho * c (Pa.s/m)
        Z = rho * c

        return rho, c, Z, alpha_0
    
    
    def aplicar_microestrutura_estocastica(self, Z: np.ndarray, volume_hu: np.ndarray):
        """
        Insere a variação de impedância sub-resolução.
        Ajustado com maior ganho estocástico no parênquima para melhorar a ecomorfologia.
        """
        sigma_map = np.zeros_like(volume_hu, dtype=np.float32)
        
        # Ignora o ar para não estragar a escala
        mascara_tecido = volume_hu > -500 
        
        # Aumentamos o contraste do espalhamento (Speckle)
        sigma_map[(volume_hu < -30) & mascara_tecido] = 0.05       # Adiposo / Gordura
        sigma_map[(volume_hu >= -30) & (volume_hu <= 150)] = 0.12  # Fibroglandular / Parênquima
        sigma_map[volume_hu > 150] = 0.02                          # Calcificações / Estruturas densas

        ruido_espalhamento = np.random.normal(0.0, sigma_map, size=Z.shape).astype(np.float32)
        return Z * (1.0 + ruido_espalhamento)

    def gerar_psf_transdutor(self, dx_mm: float, dz_mm: float):
        """
        Calcula a Point Spread Function (PSF) 2D do transdutor linear.
        """
        sigma_x = 1.2 / dx_mm                              # Resolução Lateral
        sigma_z = (self.lambda_m * 1000) / (2.0 * dz_mm)  # Resolução Axial

        sz_x = int(6 * sigma_x) | 1
        sz_z = int(6 * sigma_z) | 1
        x = np.linspace(-3*sigma_x, 3*sigma_x, sz_x)
        z = np.linspace(-3*sigma_z, 3*sigma_z, sz_z)
        xx, zz = np.meshgrid(x, z)

        envelope = np.exp(-0.5 * ((xx/sigma_x)**2 + (zz/sigma_z)**2))
        portadora = np.cos(2 * np.pi * (zz / sigma_z))
        psf = envelope * portadora
        return psf / np.sum(np.abs(psf))

    def simular_imagem_bmode_2d(self, Z_slice: np.ndarray, psf: np.ndarray):
        """
        Calcula o gradiente de reflexão, realiza a convolução com a PSF,
        extrai o envelope de rádio-frequência (RF) e aplica compressão logarítmica.
        """
        # Gradiente espacial no eixo axial (Z) -> Coeficiente de Reflexão R
        Coef_reflexao = np.gradient(Z_slice, axis=0) / (2.0 * Z_slice + 1e-6)

        # Sinal de RF
        sinal_rf = ndimage.convolve(Coef_reflexao, psf, mode='reflect')

        # Detecção de Envelope via Transformada de Hilbert
        envelope = np.abs(hilbert(sinal_rf, axis=0))

        # Compensação de Ganho em Profundidade (TGC)
        profundidade = np.linspace(0, 1, envelope.shape[0])[:, None]
        curva_tgc = np.exp(profundidade * 1.0)
        envelope_com_tgc = envelope * curva_tgc

        # Normalização com base no percentil 99.5
        p_max = np.percentile(envelope_com_tgc, 99.5) + 1e-12
        envelope_norm = envelope_com_tgc / p_max

        # Compressão Logarítmica (B-Mode Dynamic Range = 40 dB)
        bmode = 20 * np.log10(envelope_norm)
        return np.clip(bmode, -40, 0)


# --- CLASSE PARA SELEÇÃO DE ROI FREEHAND ---

class SelecionadorROIFreehand:
    """
    Interface gráfica interativa para desenhar uma ROI freehand com o mouse.
    """
    def __init__(self, fatia_2d: np.ndarray):
        self.fatia_2d = fatia_2d
        self.mascara_2d = np.zeros_like(fatia_2d, dtype=bool)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.imshow(fatia_2d, cmap='bone', vmin=-150, vmax=200)
        self.ax.set_title("DESENHE A ROI COM O MOUSE:\nClique, segure e contorne a mama. Solte para fechar.", 
                          fontsize=11, fontweight='bold', color='darkred')
        self.ax.axis('off')

        # Conecta o seletor Lasso
        self.lasso = LassoSelector(self.ax, onselect=self.onselect, props=dict(color='red', linewidth=2))
        plt.tight_layout()
        plt.show()

    def onselect(self, verts):
        """
        Callback chamado quando o usuário encerra o desenho do laço.
        """
        ny, nx = self.fatia_2d.shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        pix = np.vstack((x.flatten(), y.flatten())).T

        path = Path(verts)
        self.mascara_2d = path.contains_points(pix).reshape((ny, nx))
        plt.close(self.fig)


def gerar_mascara_3d_a_partir_de_roi_freehand(volume_shape: tuple, mascara_2d: np.ndarray) -> np.ndarray:
    """
    Extrui a máscara 2D desenhada pelo usuário para todas as fatias do volume 3D 
    e aplica uma suavização gaussiana nas bordas (Apodização / Soft Edge).
    """
    num_fatias = volume_shape[0]
    mascara_3d = np.zeros(volume_shape, dtype=np.float32)

    # Aplica a mesma ROI 2D ao longo de todas as fatias Z
    for i in range(num_fatias):
        mascara_3d[i, :, :] = mascara_2d.astype(np.float32)

    # Suaviza a borda cortada para evitar transições secas que geram artefato no US
    mascara_suave = ndimage.gaussian_filter(mascara_3d, sigma=(0, 2.0, 2.0))
    return mascara_suave


# --- FUNÇÕES DE VISUALIZAÇÃO INTERATIVA ---

def visualizar_fatiador_interativo(volume_3d: np.ndarray):
    """
    Janela interativa para navegar pelas fatias do volume 3D.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)
    
    fatia_inicial = volume_3d.shape[0] // 2
    img_display = ax.imshow(volume_3d[fatia_inicial, :, :], cmap='gray', vmin=-40, vmax=0)
    ax.set_title(f"Fatia {fatia_inicial + 1}/{volume_3d.shape[0]}")
    ax.axis('off')

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Fatia', 0, volume_3d.shape[0] - 1, valinit=fatia_inicial, valfmt='%d')

    def update(val):
        idx = int(slider.val)
        img_display.set_data(volume_3d[idx, :, :])
        ax.set_title(f"Fatia {idx + 1}/{volume_3d.shape[0]}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def visualizar_volume_3d_pyvista(volume_3d: np.ndarray, spacing: tuple, mascara_roi_3d: np.ndarray = None):
    """
    Renderiza o volume 3D em ambiente tridimensional interativo com PyVista,
    removendo completamente as regiões fora da ROI.
    """
    grid = pv.ImageData()
    grid.dimensions = np.array(volume_3d.shape) + 1
    grid.spacing = spacing[::-1]
    
    # Se uma máscara foi fornecida, zeramos o fundo para transparência total
    if mascara_roi_3d is not None:
        volume_display = np.where(mascara_roi_3d > 0.1, volume_3d, -100.0)
    else:
        volume_display = volume_3d

    grid.cell_data["Ultrassom_BMode"] = volume_display.flatten(order="F")

    plotter = pv.Plotter()
    
    # Define opacidade personalizada: valores <= -50dB ficam 100% transparentes (opacidade = 0)
    plotter.add_volume(
        grid, 
        scalars="Ultrassom_BMode", 
        cmap="gray", 
        opacity=[0.0, 0.0, 0.2, 0.6, 0.9], # Mapeamento de transparência (os dois primeiros pontos zeram o fundo)
        clim=[-50, 0]
    )
    plotter.add_axes()
    plotter.show_grid()
    plotter.show(title="Volume 3D Ultrassônico (Apenas Interior da ROI)")

# --- MAIN ---
if __name__ == "__main__":
    caminho_serie_dicom = r"C:\Users\TOMMUS\Downloads\QIN-BREAST_2015-09-04\qin_breast\QIN-BREAST-01-0002\1992-01-24-BREAST PRONE-12000\2-CTAC-14691"

    print("Inicializando a Conversão de TC para ultrassom...")
    pipeline = PipelineTCparaUltrassom(f0_mhz=5)

    if os.path.exists(caminho_serie_dicom):
        print("2. Carregando e reconstruindo o volume DICOM...")
        volume_hu, spacing = pipeline.carregar_serie_dicom(caminho_serie_dicom)
        num_fatias = volume_hu.shape[0]
        
        print(f"   -> Volume 3D Carregado: {volume_hu.shape} voxels (Fatias x Altura x Largura)")
        print(f"   -> Resolucao do Voxel (dx, dy, dz): {spacing} mm")

        print("3. Convertendo Unidades Hounsfield em Propriedades Acústicas...")
        rho, c, Z, alpha_0 = pipeline.calcular_propriedades_acusticas(volume_hu)

        print("4. Aplicando modelo estocástico de espalhadores (Speckle)...")
        Z_espalhado = pipeline.aplicar_microestrutura_estocastica(Z, volume_hu)

        # --- SELECAO INTERATIVA DO MODO DE RECORTE ---
        print("\n==============================================")
        print(" ESCOLHA O MODO DE PROCESSAMENTO:")
        print(" [1] Desenhar ROI Freehand (Desenho com o mouse)")
        print(" [2] Usar Imagem Inteira (Sem ROI)")
        print("==============================================")
        
        opcao = input("Digite 1 ou 2 e pressione Enter: ").strip()

        if opcao == "1":
            print("\nAbrindo janela para desenho da ROI Freehand...")
            fatia_central_hu = volume_hu[num_fatias // 2, :, :]
            seletor_roi = SelecionadorROIFreehand(fatia_central_hu)
            
            if np.any(seletor_roi.mascara_2d):
                print("   -> ROI Freehand capturada com sucesso!")
                mascara_roi_3d = gerar_mascara_3d_a_partir_de_roi_freehand(volume_hu.shape, seletor_roi.mascara_2d)
                select_roi = True
            else:
                print("   -> Nenhuma ROI desenhada. Processando imagem inteira.")
                mascara_roi_3d = np.ones_like(volume_hu, dtype=np.float32)
                select_roi = False
        else:
            print("\n   -> Processando a imagem inteira sem recorte.")
            mascara_roi_3d = np.ones_like(volume_hu, dtype=np.float32)
            select_roi = False

        # --- APLICAÇÃO DA IMPEDÂNCIA DO MEIO EXTERNO ---
        # Z_fundo = 1.54e6 MRayl (Impedância do meio de acoplamento/água)
        Z_fundo = 1.54e6  
        Z_processado = Z_espalhado * mascara_roi_3d + Z_fundo * (1.0 - mascara_roi_3d)

        print("\n Gerando volume de Ultrassom 3D fatia por fatia...")
        psf = pipeline.gerar_psf_transdutor(dx_mm=spacing[0], dz_mm=spacing[1])
        
        volume_ultrassom_3d = []
        for i in range(num_fatias):
            Z_slice_2d = Z_processado[i, :, :]
            bmode_slice_2d = pipeline.simular_imagem_bmode_2d(Z_slice_2d, psf)
            volume_ultrassom_3d.append(bmode_slice_2d)

        volume_ultrassom_3d = np.array(volume_ultrassom_3d, dtype=np.float32)
        print(f"   -> Volume 3D Ultrassônico concluído! Forma: {volume_ultrassom_3d.shape}")

        print("\n Exibindo os resultados comparativos da fatia central...")
        fatia_idx = num_fatias // 2
        hu_slice = volume_hu[fatia_idx, :, :]
        Z_slice = Z_processado[fatia_idx, :, :]
        bmode_slice = volume_ultrassom_3d[fatia_idx, :, :]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. TC Original (com contorno se houver ROI)
        im0 = axes[0].imshow(hu_slice, cmap='bone', vmin=-150, vmax=200)
        if select_roi:
            axes[0].contour(seletor_roi.mascara_2d, colors='red', linewidths=1)
            axes[0].set_title("1. Entrada: TC Original + ROI", fontsize=11, fontweight='bold')
        else:
            axes[0].set_title("1. Entrada: TC Original (Imagem Inteira)", fontsize=11, fontweight='bold')
        axes[0].set_xlabel("Eixo X (voxels)")
        axes[0].set_ylabel("Eixo Y (voxels)")
        fig.colorbar(im0, ax=axes[0], label="Unidades Hounsfield (HU)")

        # 2. Impedância Z Processada
        im1 = axes[1].imshow(Z_slice / 1e6, cmap='viridis')
        axes[1].set_title("2. Mapeamento: Impedância Acústica Z", fontsize=11, fontweight='bold')
        axes[1].set_xlabel("Eixo X (voxels)")
        fig.colorbar(im1, ax=axes[1], label="Impedância [MRayl]")

        # 3. Ultrassom B-mode Simulado
        im2 = axes[2].imshow(bmode_slice, cmap='gray', vmin=-40, vmax=0)
        axes[2].set_title("3. Saída: Ultrassom Simulado (B-mode)", fontsize=11, fontweight='bold')
        axes[2].set_xlabel("Eixo X (voxels)")
        fig.colorbar(im2, ax=axes[2], label="Faixa Dinâmica (dB)")

        plt.tight_layout()
        plt.show()

        print("Abrindo fatiador interativo...")
        visualizar_fatiador_interativo(volume_ultrassom_3d)

        print("Renderizando volume 3D com PyVista...")
        visualizar_volume_3d_pyvista(volume_ultrassom_3d, spacing)

    else:
        print(f"Erro: O caminho especificado não existe:\n{caminho_serie_dicom}")
    caminho_serie_dicom = r"C:\Users\TOMMUS\Downloads\QIN-BREAST_2015-09-04\qin_breast\QIN-BREAST-01-0005\1992-03-20-BREAST PRONE-33402\2-CTAC-00000"

   