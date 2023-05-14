import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io

# Carrega a imagem
img = io.imread('footBall_orig.jpg', as_gray=True)

# Calcula a Transformada Rápida de Fourier (FFT)
fft_img = np.fft.fft2(img)

# Calcula a magnitude do espectro
magnitude_spectrum = np.abs(fft_img)

# Aplica o Filtro Gaussiano passa-baixa
filtered_img = ndimage.gaussian_filter(img, sigma=3)

# Calcula a Transformada Rápida de Fourier (FFT) da imagem filtrada
fft_filtered_img = np.fft.fft2(filtered_img)

# Calcula a magnitude do espectro da imagem filtrada
magnitude_spectrum_filtered = np.abs(fft_filtered_img)

# Mostra as imagens e os espectros
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Imagem Original')

axs[0, 1].imshow(magnitude_spectrum, cmap='gray')
axs[0, 1].set_title('Espectro de Fourier')

axs[1, 0].imshow(filtered_img, cmap='gray')
axs[1, 0].set_title('Imagem Filtrada')

axs[1, 1].imshow(magnitude_spectrum_filtered, cmap='gray')
axs[1, 1].set_title('Espectro de Fourier Filtrado')

plt.show()
