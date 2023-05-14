import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import ndimage

# Carregando a imagem
img = plt.imread('footBall_orig.jpg')

# Convertendo para escala de cinza
img_gray = np.mean(img, axis=-1)

# Realizando padding para uma imagem com dimensões potência de 2
n = 2**np.ceil(np.log2(img_gray.shape)).astype(int)
img_padded = np.zeros((n[0], n[1]))
img_padded[:img_gray.shape[0], :img_gray.shape[1]] = img_gray

# Calculando a Transformada de Fourier 2D
fft_img = fftpack.fft2(img_padded)

# Calculando a magnitude do espectro
magnitude_spectrum = np.abs(fft_img)

# Aplicando o filtro gaussiano passa-baixa
sigma = 30
filt = ndimage.gaussian_filter(img_padded, sigma=sigma)

# Calculando a Transformada de Fourier 2D do filtro
fft_filt = fftpack.fft2(filt)

# Calculando a magnitude do espectro filtrado
magnitude_spectrum_filt = np.abs(fft_filt)

# Plotando as imagens e espectros
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0, 0].imshow(img_gray, cmap='gray')
axs[0, 0].set_title('Imagem Original')
axs[0, 1].imshow(magnitude_spectrum, cmap='gray')
axs[0, 1].set_title('Espectro de Fourier')
axs[1, 0].imshow(filt, cmap='gray')
axs[1, 0].set_title('Imagem Filtrada')
axs[1, 1].imshow(magnitude_spectrum_filt, cmap='gray')
axs[1, 1].set_title('Espectro de Fourier Filtrado')
for ax in axs.flat:
    ax.axis('off')
plt.show()
