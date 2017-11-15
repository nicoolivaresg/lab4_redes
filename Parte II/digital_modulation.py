import numpy as np
from scipy.io.wavfile import read,write
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.integrate import cumtrapz
import random

#### Constantes #####

GRAPH_DIR = "graph/" 
AUDIO_DIR = "audio/"

TIMEXLABEL = "time"
FREQXLABEL = "frequency"
AMPLITUDEYLABEL = "amplitude"

AUDIO_NAME = "handel"

#### Funciones #####


# Funcion que usa el read de scipy.io.wavfile para abrir un archivo .wav y obtener la
# frecuencia y la informacion del sonido, esta funcion ademas obtiene un vector de tiempo
# dependiendo de la canidad de datos y la frecuencia del audio.
#
# Entrada:
# 	filename	- Nombre del archivo por abrir.
#
# Salida:
#	frecuencia	- Numero entero con la frecuencia de muestreo del audio en [Hz].
#	datos		- Arreglo numpy con los datos obtenidos por la lectura del archivo.
#	tiempos		- Arreglo de floats con el tiempo en segundos para cada dato de 'datos'.
def load_wav_audio(filename):
	frecuencia, datos = read(filename)
	n = len(datos)
	Ts = n / frecuencia; # Intervalo de tiempo
	tiempos = np.linspace(0, Ts, n) # Tiempo en segundos para cada dato de 'datos'
	return (frecuencia, datos, tiempos)

# Proceso que crea una figura que tiene 3 graficos haciendo uso de la funcion subplot()
# que trae matplotlib.
# 
# Entrada:
#	XVector		- Vector comun x para los 3 graficos
# 	YVector1 	- Vector de valores y para el grafico de más arriba
# 	YVector2 	- Vector de valores y para el grafico del medio
#	YVector3 	- Vector de valores y para el grafico de mas abajo
#	tipo 	 	- Tipo de modulacion que usa esta figura 
#				  	(se usa para crear el nombre de la figura)
#	percentage 	- Porcentaje de modulacion que se usa en esta figura 
#					(se usa para crear el nombre de la figura).
def triple_subplot(XVector, YVector1, YVector2, YVector3, tipo ="AM", percentage=15):
	plt.subplot(311)
	plt.title("Modulación AM de la señal", fontsize=12)
	plt.plot(XVector, YVector1, linewidth=0.4)
	plt.ylabel("Amplitud")

	plt.subplot(312)
	plt.plot(XVector, YVector2, linewidth=0.4)
	plt.ylabel("Amplitud")

	plt.subplot(313)
	plt.xlabel("Tiempo[s]")
	plt.ylabel("Amplitud")
	plt.plot(XVector, YVector3, linewidth=0.4)
	plt.savefig(GRAPH_DIR + tipo + str(percentage) + ".png", bbox_inches='tight')
	plt.clf()
	plt.close('all')


def ASK_modulation(bitTime, samplingRate, A, B, carrierFrec, signal):
	n = len(signal)
	# Vector de tiempo
	time = np.linspace(0, n, bitTime)
	# Portadora para 0
	C0 = A * np.cos(2 * np.pi * carrierFrec * time)
	# Portadora para 1
	C1 = B * np.cos(2 * np.pi * carrierFrec * time)
	# Arreglo para almazenar resultados de los bits reconocidos de la señal
	y = []
	for bit in signal:
		if bit:
			y.extend(C0)
		else:
			y.extend(C1)
	return np.array(y)
	

def preProcessSignal(x):
	for num in x:
		binario = bin(num)
		
		if binario[0] == '-':
			print(binario, len(binario))	
		

	return 0

def processFile(path):
	samplingRate, signal, timeSignal = load_wav_audio(path)

	#print(samplingRate, signal, timeSignal)
	#print(len(signal)/samplingRate)
	print(preProcessSignal(signal))

processFile('handel.wav')