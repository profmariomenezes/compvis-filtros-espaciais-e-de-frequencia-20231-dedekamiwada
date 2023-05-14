import numpy as np
import matplotlib.pyplot as plt

# Criar imagem 30x30 com um retângulo de 5:24 em x e de 13:17 em y
img = np.zeros((30,30))
img[13:18, 5:25] = 1

# Calcular Transformada Rápida de Fourier (FFT2)
fft_img = np.fft.fft2(img)

# Calcular a magnitude do espectro da imagem original
mag_spectrum = np.abs(fft_img)

# Padding da imagem com zeros para aumentar a resolução
padded_img = np.pad(img, ((113, 113), (113, 113)), 'constant')
padded_fft_img = np.fft.fft2(padded_img)

# Calcular a magnitude do espectro da imagem com padding
padded_mag_spectrum = np.abs(padded_fft_img)

# Shift na Transformada, colocando a origem no centro da imagem
shifted_fft_img = np.fft.fftshift(padded_fft_img)

# Transformação logarítmica para exibir mais detalhes do espectro
log_mag_spectrum = np.log(np.abs(shifted_fft_img))

# Plotar as imagens resultantes
plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(mag_spectrum, cmap = 'gray')
plt.title('Magnitude do Espectro - sem padding'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(padded_mag_spectrum, cmap = 'gray')
plt.title('Magnitude do Espectro - com padding'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(log_mag_spectrum, cmap = 'gray')
plt.title('Magnitude do Espectro - shift e logarítmico'), plt.xticks([]), plt.yticks([])
plt.show()
