import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

# Criação da imagem
img = np.zeros((30,30))
img[13:18, 5:25] = 1

# Cálculo da Transformada de Fourier e Magnitude do Espectro
fft_img = fft2(img)
magnitude_spectrum = np.abs(fft_img)

# Padding da imagem
padded_img = np.pad(img, ((113, 113), (113, 113)), 'constant')
padded_fft_img = fft2(padded_img)
padded_magnitude_spectrum = np.abs(padded_fft_img)

# Diferenças dos espectros
print("Tamanho da imagem original:", img.shape)
print("Tamanho da imagem com padding:", padded_img.shape)
print("Tamanho do espectro da imagem original:", magnitude_spectrum.shape)
print("Tamanho do espectro da imagem com padding:", padded_magnitude_spectrum.shape)

# Shift da Transformada
shifted_fft_img = fftshift(padded_fft_img)
shifted_magnitude_spectrum = np.abs(shifted_fft_img)

# Transformação logarítmica
log_magnitude_spectrum = np.log(shifted_magnitude_spectrum)

# Plot dos resultados
plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Espectro sem Padding'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(shifted_magnitude_spectrum, cmap = 'gray')
plt.title('Espectro com Padding e Shift'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(log_magnitude_spectrum, cmap = 'gray')
plt.title('Espectro com Padding, Shift e Transformação Logarítmica'), plt.xticks([]), plt.yticks([])
plt.show()
