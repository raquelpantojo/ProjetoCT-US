"""
ETAPA 1 - Corte da mama direita e da mama esquerda em uma imagem de TC
======================================================================
Projeto CT -> US (GIIMUS / FFCLRP-USP)

O que este programa faz:
  1. Le a serie de TC (arquivos DICOM).
  2. Mostra uma fatia.
  3. Separa o corpo do ar usando um limiar de HU.
  4. Corta a mama direita e a mama esquerda usando os parametros que VOCE escolhe.
  5. Salva uma figura com o resultado.

Como usar:
  - Instale as bibliotecas (uma vez so):   pip install SimpleITK numpy matplotlib
  - Rode o programa:                        python Etapa1_corte_mamas.py
  - Mude UM parametro por vez no bloco "PARAMETROS", rode de novo e observe o que mudou.
  - Anote no caderno de laboratorio: o que mudou, o valor usado e o que aconteceu.

Lembretes:
  - A imagem e uma matriz de numeros. Cada numero e um valor em HU (Hounsfield):
        ar = -1000     gordura ~ -100     agua = 0     glandula ~ +40     osso > +300
  - volume[fatia, linha, coluna]: a "linha" cresce de cima para baixo e a "coluna" da esquerda
    para a direita da TELA.
  - A paciente esta de brucos e a imagem e reorientada: as mamas ficam na parte de CIMA.
  - Na tela, a mama DIREITA da paciente aparece do lado ESQUERDO (convencao radiologica).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

# =========================== PARAMETROS ===========================
# (mude aqui; nao precisa mexer no resto do programa nas primeiras tarefas)

PASTA_DICOM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2-CTAC-14691")

FATIA = 43            # qual fatia mostrar (0 a 82). Tarefa 1: teste 20, 43, 62.
LIMIAR_CORPO = -400   # HU acima disso e considerado "corpo". Tarefa 2: teste -900, -400, 0.

COLUNA_MEIO = 267     # coluna que divide direita/esquerda (entre as mamas). Tarefa 3.
LINHA_TOPO = 205      # primeira linha do corte (acima do mamilo). Tarefa 4.
LINHA_FUNDO = 272     # ultima linha do corte (base da mama, antes das costelas). Tarefa 4.
COLUNA_INICIO = 155   # onde comeca a mama direita (lado esquerdo da tela). Tarefa 4.
COLUNA_FIM = 372      # onde termina a mama esquerda (lado direito da tela). Tarefa 4.

JANELA_MIN, JANELA_MAX = -200, 300   # faixa de HU mostrada na tela (contraste). Tarefa 5.
# ==================================================================


def carregar_serie(pasta):
    """Le todos os arquivos DICOM da pasta e devolve o volume (em HU) e o tamanho do voxel (mm)."""
    leitor = sitk.ImageSeriesReader()
    arquivos = leitor.GetGDCMSeriesFileNames(pasta)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum DICOM encontrado em: {pasta}")
    leitor.SetFileNames(arquivos)
    imagem = sitk.DICOMOrient(leitor.Execute(), "LPS")      # deixa as mamas na parte de cima
    volume = sitk.GetArrayFromImage(imagem).astype(np.float32)
    volume = np.maximum(volume, -1000)                      # valores fora do campo viram ar
    return volume, imagem.GetSpacing()


# ---- 1. Leitura ----
volume, espacamento = carregar_serie(PASTA_DICOM)
print("Formato do volume (fatias, linhas, colunas):", volume.shape)
print("Tamanho do voxel em mm (coluna, linha, fatia):", espacamento)

# ---- 2. Uma fatia ----
fatia = volume[FATIA]
print(f"Fatia {FATIA}: HU minimo = {fatia.min():.0f}, maximo = {fatia.max():.0f}, medio = {fatia.mean():.0f}")

# ---- 3. Corpo x ar ----
corpo = fatia > LIMIAR_CORPO          # matriz de Verdadeiro/Falso
print(f"Pixels de corpo na fatia: {corpo.sum()}  ({100 * corpo.mean():.1f}% da imagem)")

# ---- 4. Cortes ----
mama_direita = fatia[LINHA_TOPO:LINHA_FUNDO, COLUNA_INICIO:COLUNA_MEIO]
mama_esquerda = fatia[LINHA_TOPO:LINHA_FUNDO, COLUNA_MEIO:COLUNA_FIM]
print("Tamanho do corte da mama direita  (linhas, colunas):", mama_direita.shape)
print("Tamanho do corte da mama esquerda (linhas, colunas):", mama_esquerda.shape)

dx, dy, dz = espacamento
print(f"Em mm: direita {mama_direita.shape[1] * dx:.0f} x {mama_direita.shape[0] * dy:.0f} mm, "
      f"esquerda {mama_esquerda.shape[1] * dx:.0f} x {mama_esquerda.shape[0] * dy:.0f} mm")

# ---- 5. Figura ----
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

ax[0, 0].imshow(fatia, cmap="gray", vmin=JANELA_MIN, vmax=JANELA_MAX)
ax[0, 0].axvline(COLUNA_MEIO, color="yellow", linestyle="--")
ax[0, 0].add_patch(plt.Rectangle((COLUNA_INICIO, LINHA_TOPO), COLUNA_MEIO - COLUNA_INICIO,
                                 LINHA_FUNDO - LINHA_TOPO, fill=False, edgecolor="red"))
ax[0, 0].add_patch(plt.Rectangle((COLUNA_MEIO, LINHA_TOPO), COLUNA_FIM - COLUNA_MEIO,
                                 LINHA_FUNDO - LINHA_TOPO, fill=False, edgecolor="cyan"))
ax[0, 0].set_title(f"Fatia {FATIA}: vermelho = mama direita, ciano = mama esquerda")
ax[0, 0].set_xlabel("coluna"); ax[0, 0].set_ylabel("linha")

ax[0, 1].imshow(corpo, cmap="gray")
ax[0, 1].set_title(f"Corpo (HU > {LIMIAR_CORPO})")

ax[1, 0].imshow(mama_direita, cmap="gray", vmin=JANELA_MIN, vmax=JANELA_MAX)
ax[1, 0].set_title("Corte: mama direita")
ax[1, 1].imshow(mama_esquerda, cmap="gray", vmin=JANELA_MIN, vmax=JANELA_MAX)
ax[1, 1].set_title("Corte: mama esquerda")

plt.tight_layout()
plt.savefig("etapa1_resultado.png", dpi=100)
print("Figura salva em etapa1_resultado.png")
plt.show()

# ======================= TAREFAS (uma por vez) =======================
# Tarefa 1  Mude FATIA para 20, 43 e 62. Em quais fatias as mamas aparecem? Qual a primeira e a ultima?
# Tarefa 2  Mude LIMIAR_CORPO para -900, -400 e 0. O que aparece e o que some na imagem "Corpo"? Por que?
# Tarefa 3  Mude COLUNA_MEIO ate a linha amarela ficar exatamente no sulco entre as mamas.
# Tarefa 4  Ajuste LINHA_TOPO, LINHA_FUNDO, COLUNA_INICIO e COLUNA_FIM para o retangulo conter a mama
#           inteira, com pouca sobra. Os valores servem para outras fatias? Teste.
# Tarefa 5  Mude JANELA_MIN/JANELA_MAX para (-150, 50) e (-1000, 1000). Qual mostra melhor gordura x glandula?
#
# --- A partir daqui voce vai ESCREVER codigo novo (peca ajuda quando travar) ---
# Tarefa 6  Calcule e imprima o HU medio dentro de cada corte:  print(mama_direita.mean())
#           Depois, so dos pixels de tecido:  mama_direita[mama_direita > LIMIAR_CORPO].mean()
# Tarefa 7  Faca um histograma dos HU do corte:  plt.hist(mama_direita.ravel(), bins=100)
#           Onde estao os picos de ar, gordura e glandula?
# Tarefa 8  Crie um novo parametro LIMIAR_GLANDULA = -30 e calcule a porcentagem de tecido glandular
#           (pixels com HU > LIMIAR_GLANDULA dividido pelos pixels de tecido) em cada mama.
# Tarefa 9  Use um laco "for" para repetir o corte em todas as fatias de FATIA_INICIAL a FATIA_FINAL
#           (dois parametros novos) e guarde os cortes em uma lista.
# Tarefa 10 Salve o volume cortado de cada mama com np.save("mama_direita.npy", ...) e confira o
#           formato (fatias, linhas, colunas) ao abrir de novo com np.load.
# =====================================================================
