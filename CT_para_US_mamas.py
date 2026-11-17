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
     O TRANSDUTOR e posicionado pelo usuario (ver TRANSDUTORES no __main__): sonda linear no
     plano axial, com centro da face, angulo do feixe, largura e profundidade. A posicao pode
     ser clicada na TC ou digitada em mm; sem posicao, a sonda vai para o mamilo, com o feixe
     de anterior para posterior. A sonda fica fixa e a varredura e feita fatia a fatia na
     direcao S-I (como num sistema motorizado), em agua.
  6. Grava, para cada transdutor e cada mama, novas series DICOM:
       saida/<mama>/US_<nome>/ -> ultrassom simulado no plano da sonda (8 bits, 0 = -faixa
                                  dinamica, 255 = 0 dB), com a orientacao da sonda no DICOM
       saida/<mama>/CT/  -> TC recortada (HU, int16; fora da mama = -1000 se MASCARAR_CT)
     As series ficam no mesmo estudo e no mesmo Frame of Reference da TC original, com posicao
     e orientacao (ImagePositionPatient / ImageOrientationPatient) corretas: em visualizadores
     como o 3D Slicer, US e TC se sobrepoem.

Convencao de lados: em LPS a coluna cresce para a ESQUERDA do paciente. Na tela (convencao
radiologica) a mama direita aparece a esquerda da imagem.

Convencao do transdutor (mesmas coordenadas das figuras): x (mm) cresce para a direita da tela
(esquerda da paciente) e y (mm) para baixo (posterior), a partir do canto da imagem.
centro_mm = (x, y) do centro da face da sonda; angulo_graus = 0 -> feixe para +y (anterior ->
posterior); angulo > 0 gira o feixe para +x (para a esquerda da paciente).

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
from TransformCTUS_kutter import SimuladorKutter, gradiente_deriche_3d  # noqa: E402

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
    return dict(meio=meio, corte=corte, mascaras=mascaras, fatias=fatias,
                mamilos={"mama_direita": mamilo_dir, "mama_esquerda": mamilo_esq}, perfil=F)


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
    ax[0].axvline((sep["meio"] + 0.5) * dx, color="yellow", ls="--", lw=1, label="linha media")
    if np.isfinite(sep["corte"][k_meio]):
        ax[0].axhline((sep["corte"][k_meio] + 0.5) * dy, color="orange", ls=":", lw=1, label="limite posterior")
    ax[0].set_title(f"Fatia {k_meio}: vermelho = mama direita, ciano = mama esquerda", fontsize=11)
    ax[0].set_xlabel("x (mm) -> esquerda do paciente"); ax[0].set_ylabel("y (mm) -> posterior")
    ax[0].legend(loc="lower right", fontsize=9)
    ys = [y for y in (sep["perfil"].min() - 15, np.nanmax(np.where(np.isfinite(sep["corte"]), sep["corte"], np.nan)) + 25)]
    ax[0].set_ylim(min(ys[1] * dy, ny * dy), max(ys[0] * dy, 0))

    # projecao frontal (coronal) da extensao das mamas
    proj = np.zeros(vol.shape[::2], dtype=np.int8)            # (Z, X)
    for i, lado in enumerate(LADOS, start=1):
        proj[sep["mascaras"][lado].any(axis=1)] = i
    ax[1].imshow(proj, cmap="viridis", origin="lower", aspect="equal",
                 extent=[0, nx * dx, 0, vol.shape[0] * spacing[2]])
    ax[1].axvline((sep["meio"] + 0.5) * dx, color="yellow", ls="--", lw=1)
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
def vetores_transdutor(angulo_graus):
    """Direcao do feixe d e direcao lateral u (unitarios, em (x, y) da imagem)."""
    a = np.deg2rad(angulo_graus)
    d = np.array([np.sin(a), np.cos(a)])          # profundidade
    u = np.array([np.cos(a), -np.sin(a)])         # lateral (u x d = +z: orientacao direita)
    return u, d


def posicao_automatica(mascara, k, spacing, folga_mm=3.0):
    """Centro da sonda sobre o ponto mais anterior (mamilo) da mama na fatia k, com folga em agua."""
    dx, dy, _ = spacing
    ys, xs = np.where(mascara[k])
    i = np.argmin(ys)
    return ((xs[i] + 0.5) * dx, (ys[i] + 0.5) * dy - folga_mm)


def escolher_transdutor_clique(vol, mascara, k, spacing, largura_mm, profundidade_mm, titulo=""):
    """
    Mostra a fatia k e pede dois cliques: 1) centro da face do transdutor (na agua, junto a pele);
    2) um ponto na direcao do feixe (dentro da mama). Devolve (centro_mm, angulo_graus).
    """
    dx, dy, _ = spacing
    ny, nx = vol.shape[1:]
    ys, xs = np.where(mascara[k])
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(vol[k], cmap="bone", vmin=-200, vmax=300, extent=[0, nx * dx, ny * dy, 0])
    ax.contour(mascara[k], [0.5], colors="cyan", linewidths=1, extent=[0, nx * dx, 0, ny * dy])
    m = 40.0
    ax.set_xlim(xs.min() * dx - m, xs.max() * dx + m)
    ax.set_ylim(ys.max() * dy + m, ys.min() * dy - m)
    ax.set_title(f"{titulo} - fatia {k}\nClique 1: centro da face do transdutor (na agua, junto a pele)\n"
                 "Clique 2: um ponto na direcao do feixe (dentro da mama)", fontsize=10, color="darkred")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) < 2:
        raise RuntimeError("Sao necessarios dois cliques para posicionar o transdutor.")
    (x0, y0), (x1, y1) = pts
    angulo = float(np.rad2deg(np.arctan2(x1 - x0, y1 - y0)))
    return (float(x0), float(y0)), angulo


def coordenadas_transdutor(spacing, centro_mm, angulo_graus, largura_mm, profundidade_mm, dz, dx_lat):
    """
    Grade do plano da sonda: ponto(s, a) = centro + a*u + s*d, s = profundidade, a = lateral.
    Devolve (coords [2, ns, na] em indices (linha, coluna) da TC, prof, lat, u, d).
    """
    dx, dy, _ = spacing
    u, d = vetores_transdutor(angulo_graus)
    ns = int(round(profundidade_mm / dz))
    na = int(round(largura_mm / dx_lat))
    prof = np.arange(ns) * dz
    lat = (np.arange(na) - (na - 1) / 2.0) * dx_lat
    S, A = np.meshgrid(prof, lat, indexing="ij")
    X = centro_mm[0] + A * u[0] + S * d[0]
    Y = centro_mm[1] + A * u[1] + S * d[1]
    return np.stack([Y / dy - 0.5, X / dx - 0.5]), prof, lat, u, d


def simular_transdutor(vol, corpo, mascara, inclusao, spacing, sim, centro_mm, angulo_graus,
                       largura_mm, profundidade_mm, fatias, limitar_a_mama=False,
                       ecogenicidade_inclusao=0.1, rotulo=""):
    """
    Simula a imagem da sonda posicionada pelo usuario em cada fatia de 'fatias'.
    A fisica usa toda a anatomia no campo de visao (parede toracica, costelas...);
    com limitar_a_mama=True, os pixels fora da mama sao zerados no fim.
    Devolve (vol_db [fatias, prof, lateral], prof, lat, u, d).
    """
    dx, dy, dz = spacing
    k0, k1 = int(fatias[0]), int(fatias[-1]) + 1
    coords, prof, lat, u, d = coordenadas_transdutor(spacing, centro_mm, angulo_graus, largura_mm,
                                                     profundidade_mm, sim.dz_fino, sim.dx_fino)
    # recorte da TC que contem o campo de visao (+ margem), para acelerar
    mg = 8
    r0 = max(int(np.floor(coords[0].min())) - mg, 0); r1 = min(int(np.ceil(coords[0].max())) + mg, vol.shape[1])
    c0 = max(int(np.floor(coords[1].min())) - mg, 0); c1 = min(int(np.ceil(coords[1].max())) + mg, vol.shape[2])
    if r1 <= r0 or c1 <= c0:
        raise ValueError("O campo de visao do transdutor esta fora da imagem.")
    sl = (slice(k0, k1), slice(r0, r1), slice(c0, c1))
    vol_c, corpo_c, masc_c = vol[sl], corpo[sl], mascara[sl]
    coords_c = coords - np.array([r0, c0])[:, None, None]

    t0 = time.time()
    _, m, _ = sim.preprocessar_volume(vol_c, corpo_c, spacing, roi_3d=masc_c.astype(np.float32))
    if inclusao is not None and inclusao[sl].any():
        # Inclusao (tumor): hipoecoica no mapa de espalhamento; a borda continua refletindo
        # porque a impedancia vem do HU da TC.
        m = np.where(inclusao[sl], np.float32(ecogenicidade_inclusao), m)
    # termo lambertiano |r.n| com r = direcao do feixe da sonda (pode ser obliqua)
    hu = np.where(corpo_c, vol_c, 0.0).astype(np.float32)
    gz, gy, gx = gradiente_deriche_3d(hu, (dz, dy, dx), sim.escala_deriche_mm)
    gmag = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)
    rn = ((np.abs(gx * d[0] + gy * d[1]) / (gmag + 1e-6)) ** sim.p_lambert).astype(np.float32)

    uns = np.ones(vol_c.shape[1:], np.float32)
    saida = None
    for i in range(vol_c.shape[0]):
        Ius, _, _ = sim.simular_fatia(vol_c[i], corpo_c[i], uns, rn[i], m[i], coords_c, spacing)
        Ius = Ius[::2]
        if limitar_a_mama:
            dentro = ndimage.map_coordinates(masc_c[i].astype(np.float32), coords_c[:, ::2], order=1) > 0.5
            Ius = np.where(dentro, Ius, 0.0)
        if saida is None:
            saida = np.zeros((vol_c.shape[0],) + Ius.shape, np.float32)
        saida[i] = Ius
        print(f"\r   -> {rotulo}: fatia {i + 1}/{vol_c.shape[0]}", end="", flush=True)
    print(f"  ({time.time() - t0:.0f} s)")
    return sim.comprimir_log(saida), prof[::2], lat, u, d


# ====================================================================== #
# 6. Gravacao DICOM
# ====================================================================== #
_RAIZ_UID = "1.2.826.0.1.3680043.2.1125."     # raiz usada nos exemplos do SimpleITK/GDCM


def novo_uid():
    agora = datetime.datetime.now()
    return _RAIZ_UID + agora.strftime("%Y%m%d%H%M%S") + "." + str(agora.microsecond) + str(np.random.randint(1000, 9999))


def escrever_serie_dicom(vol, origem, espacamento, tags_base, pasta, descricao, modalidade,
                         numero_serie, extras=None, direcao=None):
    """
    Grava vol [Z, Y, X] (uint8 ou int16) como uma serie DICOM, uma fatia por arquivo.
    origem = centro do voxel [0, 0, 0] em mm (LPS). direcao = matriz 3x3 (colunas = direcoes
    dos eixos X, Y, Z da imagem em LPS); None = identidade.
    """
    os.makedirs(pasta, exist_ok=True)
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing([float(s) for s in espacamento])
    img.SetOrigin([float(o) for o in origem])
    D = np.eye(3) if direcao is None else np.asarray(direcao, float)
    img.SetDirection(tuple(D.ravel()))
    orient = "\\".join(f"{v:.6f}" for v in list(D[:, 0]) + list(D[:, 1]))

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
        ("0020|0037", orient),                           # Image Orientation (Patient)
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
              ecogenicidade_inclusao=0.1, fracao_suv=0.40, suv_minimo=2.5, transdutores=None,
              limitar_a_mama=False):
    os.makedirs(pasta_saida, exist_ok=True)
    print("1. Lendo a serie DICOM e reorientando para LPS...")
    img, vol, tags = carregar_serie(caminho_serie)
    spacing = img.GetSpacing()                                    # (dx, dy, dz)
    print(f"   -> {vol.shape} (fatias x linhas x colunas), voxel {tuple(round(s, 3) for s in spacing)} mm, "
          f"posicao do paciente: {tags.get('0018|5100', '?').strip()}")

    print("2. Segmentando o corpo...")
    corpo = SimuladorKutter.mascara_corpo(vol)

    print("3. Separando mama direita e esquerda...")
    sep = separar_mamas(vol, corpo, spacing, offset_corte_mm, protrusao_min_mm, meio_coluna)
    for lado in LADOS:
        f = sep["fatias"][lado]
        print(f"   -> {lado.replace('_', ' ')}: fatias {f[0]} a {f[-1]} ({f.size}), "
              f"volume {sep['mascaras'][lado].sum() * np.prod(spacing) / 1000:.0f} cm3")
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
            lado_ok = (np.arange(vol.shape[2]) < sep["meio"]) if lado == "mama_direita" else (np.arange(vol.shape[2]) >= sep["meio"])
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
            print("   Ajuste OFFSET_CORTE_MM, PROTRUSAO_MIN_MM ou MEIO_COLUNA no inicio do __main__ e rode de novo.")
            return None

    # ---------------- TC recortada e inclusao de cada mama (independe da sonda) ----------------
    nomes = {"mama_direita": "direita", "mama_esquerda": "esquerda"}
    for n_lado, lado in enumerate(LADOS):
        if not (salvar_ct or inclusoes[lado] is not None):
            continue
        k0, k1, r0, r1, c0, c1 = caixa_da_mama(sep["mascaras"][lado], spacing, meio=sep["meio"], lado=lado)
        img_c = img[c0:c1, r0:r1, k0:k1]                          # mantem a geometria fisica
        pasta_lado = os.path.join(pasta_saida, lado)
        if salvar_ct:
            ct = vol[k0:k1, r0:r1, c0:c1].copy()
            if mascarar_ct:
                ct[~sep["mascaras"][lado][k0:k1, r0:r1, c0:c1]] = -1000.0
            escrever_serie_dicom(np.round(ct).astype(np.int16), img_c.GetOrigin(), spacing, tags,
                                 os.path.join(pasta_lado, "CT"), f"TC recortada - mama {nomes[lado]}", "CT",
                                 901 + 2 * n_lado,
                                 extras={"0028|1052": "0", "0028|1053": "1", "0028|1050": "40", "0028|1051": "400"})
            print(f"   -> DICOM TC: {os.path.join(pasta_lado, 'CT')}")
        if inclusoes[lado] is not None:
            escrever_serie_dicom(inclusoes[lado][k0:k1, r0:r1, c0:c1].astype(np.uint8), img_c.GetOrigin(), spacing,
                                 tags, os.path.join(pasta_lado, "INCLUSAO"),
                                 f"Inclusao (PET {int(100 * fracao_suv)}% SUVmax) - mama {nomes[lado]}",
                                 "OT", 905 + n_lado, extras={"0028|1050": "0.5", "0028|1051": "1"})
            print(f"   -> DICOM inclusao: {os.path.join(pasta_lado, 'INCLUSAO')}")

    # ---------------- transdutores ----------------
    if not transdutores:
        transdutores = [dict(nome="T1", mama="mama_esquerda" if inclusoes["mama_esquerda"] is not None
                             else "mama_direita")]
    dx, dy, dz = spacing
    resultados = []
    for n_t, cfg in enumerate(transdutores):
        cfg = dict(cfg)
        nome_t = cfg.get("nome", f"T{n_t + 1}")
        lado = cfg.get("mama", "mama_esquerda")
        if lado not in LADOS:
            raise ValueError(f"mama deve ser uma de {LADOS}, nao '{lado}'")
        mascara = sep["mascaras"][lado]
        largura = float(cfg.get("largura_mm", 36.8))
        profundidade = float(cfg.get("profundidade_mm", 50.0))
        f_mama = np.where(mascara.any(axis=(1, 2)))[0]
        fatias = cfg.get("fatias") or (int(f_mama[0]), int(f_mama[-1]))
        fatias = (max(int(fatias[0]), 0), min(int(fatias[1]), vol.shape[0] - 1))
        alvo = inclusoes[lado] if inclusoes[lado] is not None else mascara
        k_ref = cfg.get("fatia_referencia")
        if k_ref is None:
            k_ref = int(np.argmax(alvo.sum(axis=(1, 2))))
        modo = cfg.get("modo", "manual")
        print(f"7. TRANSDUTOR {nome_t} ({lado.replace('_', ' ')}, modo {modo})")
        if modo == "clique":
            centro, angulo = escolher_transdutor_clique(vol, mascara, k_ref, spacing, largura, profundidade,
                                                        titulo=f"Transdutor {nome_t}")
        elif cfg.get("centro_mm") is not None:
            centro, angulo = tuple(cfg["centro_mm"]), float(cfg.get("angulo_graus", 0.0))
        else:
            centro, angulo = posicao_automatica(mascara, k_ref, spacing, cfg.get("folga_mm", 3.0)), \
                float(cfg.get("angulo_graus", 0.0))
        print(f"   -> centro_mm = ({centro[0]:.1f}, {centro[1]:.1f}), angulo_graus = {angulo:.1f}, "
              f"largura {largura:.1f} mm, profundidade {profundidade:.1f} mm, fatias {fatias[0]}-{fatias[1]}")
        print(f"      (para repetir sem clicar: dict(nome='{nome_t}', mama='{lado}', modo='manual', "
              f"centro_mm=({centro[0]:.1f}, {centro[1]:.1f}), angulo_graus={angulo:.1f}, "
              f"largura_mm={largura}, profundidade_mm={profundidade}, fatias={tuple(fatias)}))")

        sim = SimuladorKutter(**(parametros_simulador or {}))
        vol_db, prof, lat, u, d = simular_transdutor(
            vol, corpo, mascara, inclusoes[lado], spacing, sim, centro, angulo, largura, profundidade,
            fatias, limitar_a_mama=limitar_a_mama, ecogenicidade_inclusao=ecogenicidade_inclusao,
            rotulo=f"{nome_t}")

        # geometria DICOM: pixel (coluna i, linha j) = centro + lat[i]*u + prof[j]*d, na fatia k
        pto = lambda x_mm, y_mm, k: img.TransformContinuousIndexToPhysicalPoint((x_mm / dx - 0.5, y_mm / dy - 0.5, float(k)))
        p00 = np.array(pto(centro[0] + lat[0] * u[0] + prof[0] * d[0], centro[1] + lat[0] * u[1] + prof[0] * d[1], fatias[0]))
        ex = np.array(pto(centro[0] + (lat[0] + 1) * u[0], centro[1] + (lat[0] + 1) * u[1], fatias[0])) - \
            np.array(pto(centro[0] + lat[0] * u[0], centro[1] + lat[0] * u[1], fatias[0]))
        ey = np.array(pto(centro[0] + d[0], centro[1] + d[1], fatias[0])) - np.array(pto(centro[0], centro[1], fatias[0]))
        ez = np.array(pto(centro[0], centro[1], fatias[0] + 1)) - np.array(pto(centro[0], centro[1], fatias[0]))
        D = np.stack([ex / np.linalg.norm(ex), ey / np.linalg.norm(ey), ez / np.linalg.norm(ez)], axis=1)
        esp_us = (sim.dx_fino, sim.dz_fino * 2, dz)
        dr = sim.faixa_dinamica_db
        us8 = np.round((vol_db + dr) / dr * 255.0).astype(np.uint8)
        pasta_t = os.path.join(pasta_saida, lado, f"US_{nome_t}")
        escrever_serie_dicom(us8, p00, esp_us, tags, pasta_t,
                             f"US simulado {nome_t} (Kutter 2009) - mama {nomes[lado]}", "US", 920 + n_t,
                             extras={"0028|1050": "128", "0028|1051": "256",
                                     "0018|6030": str(sim.f0_mhz * 1e6)},
                             direcao=D)
        print(f"   -> DICOM US: {pasta_t} ({us8.shape[0]} fatias, {us8.shape[2]} x {us8.shape[1]} pixels "
              f"de {esp_us[0]:.3f} x {esp_us[1]:.3f} mm)")
        resultados.append(dict(nome=nome_t, mama=lado, centro_mm=centro, angulo_graus=angulo, largura_mm=largura,
                               profundidade_mm=profundidade, fatias=fatias, k_ref=k_ref, us_db=vol_db,
                               prof=prof, lat=lat, u=u, d=d, dr=dr))

    # ---------------- figura: sonda sobre a TC + imagem simulada ----------------
    ny, nx = vol.shape[1:]
    fig, ax = plt.subplots(2, len(resultados), figsize=(7 * len(resultados), 11), squeeze=False)
    for j, R in enumerate(resultados):
        k = int(np.clip(R["k_ref"], R["fatias"][0], R["fatias"][1]))
        c, u, d = np.array(R["centro_mm"]), R["u"], R["d"]
        W, P = R["largura_mm"], R["profundidade_mm"]
        cantos = np.array([c - W / 2 * u, c + W / 2 * u, c + W / 2 * u + P * d, c - W / 2 * u + P * d, c - W / 2 * u])
        a0 = ax[0, j]
        a0.imshow(vol[k], cmap="bone", vmin=-200, vmax=300, extent=[0, nx * dx, ny * dy, 0])
        a0.contour(sep["mascaras"][R["mama"]][k], [0.5], colors="cyan", linewidths=1, extent=[0, nx * dx, 0, ny * dy])
        if inclusoes[R["mama"]] is not None and inclusoes[R["mama"]][k].any():
            a0.contour(inclusoes[R["mama"]][k], [0.5], colors="lime", linewidths=1.2, extent=[0, nx * dx, 0, ny * dy])
        a0.plot(cantos[:, 0], cantos[:, 1], "y--", lw=1)
        a0.plot(cantos[:2, 0], cantos[:2, 1], color="orange", lw=5, solid_capstyle="butt")   # face da sonda
        a0.annotate("", xy=c + 0.35 * P * d, xytext=c, arrowprops=dict(color="orange", arrowstyle="->", lw=2))
        xs_, ys_ = cantos[:, 0], cantos[:, 1]
        a0.set_xlim(xs_.min() - 30, xs_.max() + 30); a0.set_ylim(ys_.max() + 30, ys_.min() - 30)
        a0.set_title(f"{R['nome']}: fatia {k}, centro ({c[0]:.0f}, {c[1]:.0f}) mm, angulo {R['angulo_graus']:.0f} graus")
        a0.set_xlabel("x (mm)"); a0.set_ylabel("y (mm)")
        a1 = ax[1, j]
        a1.imshow(R["us_db"][k - R["fatias"][0]], cmap="gray", vmin=-R["dr"], vmax=0,
                  extent=[R["lat"][0], R["lat"][-1], R["prof"][-1], R["prof"][0]])
        a1.set_title(f"US simulado {R['nome']} - {R['mama'].replace('_', ' ')}")
        a1.set_xlabel("lateral (mm)"); a1.set_ylabel("profundidade (mm)")
    plt.tight_layout()
    arq = os.path.join(pasta_saida, "resultado_transdutores.png")
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
    OFFSET_CORTE_MM = 10.0      # quanto alem da pele da linha media entra no recorte (base da mama)
    PROTRUSAO_MIN_MM = 10.0     # fatias em que a mama se projeta menos que isso ficam de fora
    MEIO_COLUNA = None          # coluna da linha media (None = automatico pelo sulco intermamario)
    SALVAR_CT = True            # grava tambem a TC recortada de cada mama
    MASCARAR_CT = True          # na TC gravada, fora da mama = -1000 HU
    CONFIRMAR = True            # pergunta se a separacao esta correta antes de simular
    MOSTRAR_FIGURAS = True
    PARAMETROS_SIMULADOR = dict(f0_mhz=5.0)   # qualquer parametro de SimuladorKutter (alpha_mistura, ...)
    # Inclusao a partir do PET do mesmo estudo (None = sem inclusao)
    PASTA_PET = os.path.join(PASTA_SCRIPT, "3-PET AC 3DWB-52072")
    ECOGENICIDADE_INCLUSAO = 0.1   # mapa de espalhamento na lesao (gordura ~0.2, glandula ~1): hipoecoica
    FRACAO_SUV = 0.40              # limiar de segmentacao: 40% do SUVmax
    SUV_MINIMO = 2.5               # abaixo disso considera que a mama nao tem inclusao

    # TRANSDUTOR(ES): um dicionario por posicao de sonda (podem ser varios).
    #   nome            : nome da serie (pasta US_<nome>)
    #   mama            : "mama_direita" ou "mama_esquerda"
    #   modo            : "clique" (escolhe na tela) ou "manual" (usa centro_mm/angulo_graus)
    #   centro_mm       : (x, y) do centro da face da sonda, em mm, como nos eixos das figuras
    #                     (None no modo manual = sobre o mamilo, folga_mm acima da pele)
    #   angulo_graus    : 0 = feixe de anterior para posterior; > 0 gira para a esquerda da paciente
    #   largura_mm      : largura da face (L12-5: 36.8 mm)
    #   profundidade_mm : profundidade da imagem
    #   fatias          : (primeira, ultima) fatia da varredura S-I (None = toda a mama)
    #   fatia_referencia: fatia mostrada para o clique e na figura (None = maior area da lesao/mama)
    # No modo clique o programa imprime a linha pronta para repetir a mesma posicao no modo manual.
    TRANSDUTORES = [
        dict(nome="T1", mama="mama_esquerda", modo="clique", angulo_graus=0.0,
             largura_mm=36.8, profundidade_mm=50.0, fatias=None),
        # exemplo manual:
        # dict(nome="T2", mama="mama_direita", modo="manual", centro_mm=(200.0, 315.0),
        #      angulo_graus=-20.0, largura_mm=36.8, profundidade_mm=50.0, fatias=(30, 50)),
    ]
    LIMITAR_A_MAMA = False  # False: mostra tudo que a sonda "ve" (padrao); True: zera fora da mama
    # ----------------------------------------------------------------

    processar(CAMINHO_SERIE, PASTA_SAIDA, OFFSET_CORTE_MM, PROTRUSAO_MIN_MM, MEIO_COLUNA,
              SALVAR_CT, MASCARAR_CT, CONFIRMAR, MOSTRAR_FIGURAS, PARAMETROS_SIMULADOR,
              PASTA_PET if os.path.isdir(PASTA_PET) else None, ECOGENICIDADE_INCLUSAO, FRACAO_SUV, SUV_MINIMO,
              TRANSDUTORES, LIMITAR_A_MAMA)
