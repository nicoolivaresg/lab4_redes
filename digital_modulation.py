import numpy as np
import math
from scipy.io.wavfile import read,write
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.integrate import cumtrapz
import random

#### Constantes #####
DPI = 100
FIGURE_WIDTH = 8
FIGURE_HEIGHT = 3

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
#	XVector		- Vector comun x para los 3 graficos.
# 	YVector1 	- Vector de valores y para el grafico de más arriba.
# 	YVector2 	- Vector de valores y para el grafico del medio.
#	YVector3 	- Vector de valores y para el grafico de mas abajo.
#	filename	- Nombre del archivo en el que se guarda la figura (sin extension).
##
def triple_subplot(XVector, YVector1, YVector2, YVector3, filename):
	plt.subplot(311)
	plt.title("Señal digitalizada", fontsize=12)
	plt.plot(XVector, YVector1, linewidth=0.4)
	plt.ylabel("Amplitud")

	plt.subplot(312)
	plt.title("Señal modulada por ASK", fontsize=12)
	plt.plot(XVector, YVector2, linewidth=0.4)
	plt.ylabel("Amplitud")

	plt.subplot(313)
	plt.title("Señal modulada por ASK + ruido AWGN", fontsize=12)
	plt.xlabel("Tiempo[s]")
	plt.ylabel("Amplitud")
	plt.plot(XVector, YVector3, linewidth=0.4)
	plt.savefig(GRAPH_DIR + filename + ".png", bbox_inches='tight')
	plt.clf()
	plt.close('all')

# Funcion que grafica los datos en ydata y xdata, y escribe los nombres del eje x, eje y,
# y el titulo de una figura. Esta figura la guarda en un archivo con el nombre filename.png.
#
# Entrada:
#	filename	- Nombre del archivo en donde se guarda la figura.
#	title		- Titulo de la figura.
#	ylabel		- Etiqueta del eje y.
#	xlabel		- Etiqueta del eje x.
#	ydata		- Datos del eje y.
#	xdata		- Datos del eje X, por defecto es un arreglo vacío que luego se cambia por un
#				  arreglo desde 0 hasta largo de ydata - 1
#	color		- Color de la figura en el grafico, por defecto es azul (blue).
def graficar(filename, title, ylabel, xlabel, ydata, xdata=np.array([]), color='b'):
	if xdata.size == 0:
		xdata = np.arange(len(ydata))

	plt.figure(figsize=(FIGURE_WIDTH,FIGURE_HEIGHT), dpi=DPI)
	plt.plot(xdata, ydata, color)

	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.savefig(GRAPH_DIR + filename + ".png", bbox_inches='tight')
	plt.clf()
	plt.close('all')


def ASK_modulation(bitspersec, A, B, carrierFrec, signal):
	# Vector de tiempo
	time = np.linspace(0, 1, bitspersec)
	if carrierFrec != 5310:
		# Formula del profe
		time = np.arange(0, 5/carrierFrec, 1/(carrierFrec*10.5))
	# Portadora para 1
	C1 = B * np.cos(2 * np.pi * carrierFrec * time)
	# Portadora para 0
	C0 = A * np.cos(2 * np.pi * carrierFrec * time)

	#graficar("C0","C0","amplitud","tiempo",C0,time)
	#graficar("C1","C1","amplitud","tiempo",C1,time)
	# Arreglo para almacenar resultados de los bits reconocidos de la señal
	y = []
	for bit in signal:
		if bit:
			y.extend(C1)
		else:
			y.extend(C0)
	return np.array(y)
	

def preProcessSignal(x):
	y = []
	i = 0
	for num in x:
		#Representación a binario
		if ( num - math.floor(num) ) >= 0.5:
			binario = np.binary_repr( math.ceil(num) , 16)
		else:
			binario = np.binary_repr( math.floor(num), 16)
		y.extend( list(binario) )
	z = []
	for elem in y:
		z.append(int(elem))
	return np.array(z)

def processFile(path):
	
	carrierFrec = 5310
	#bitTime = 0.1
	bitspersec = 33
	#baudRate = 6 
	cut = 1000 # Cuantos datos se cortan para de la señal digitalizada para modular
	start = 500	# Desde que dato
	end = 1600 # Hasta que dato se grafica
	A = 1
	B = 5
	samplingRate, signal, timeSignalVector = load_wav_audio(path)
	signalTime = len(signal)/samplingRate

	#Preproceso de señal, conversion a bits (se corta para acelerar los calculos)
	signalSample = signal[:cut]
	newsignal = preProcessSignal(signalSample)

	#Modulacion de la señal
	#ASKResult = ASK_modulation(bitspersec, A,B,carrierFrec, signalSample)
	timeVector = np.linspace(0, signalTime, len(newsignal))

	# Cortar la señal digitalizada 
	cutSignal = newsignal[:cut]
	# Datos usados para interpolar la señal digitalizada para que tenga la misma cantidad
	# de datos que la modulacion y se puedan graficar con el mismo vector de tiempo
	oldX = np.linspace(0, signalTime * (cut / len(newsignal)), cut)
	newX = np.linspace(0, signalTime * (cut / len(newsignal)), cut * bitspersec)

	# Modulacion de la señal digital cortada
	ASKBinary = ASK_modulation(bitspersec, A, B, carrierFrec, cutSignal)
	cutSignal = np.interp(newX, oldX, cutSignal)
	
	# FALTA AGREGAR RUIDO Y LA DEMODULACION
	# Graficar la señal digital, su modulacion y la modulacion con ruido
	#triple_subplot(timeVector[start:end], signal[start:end], ASKResult[start:end], ASKResult[start:end], "test1")	
	triple_subplot(timeVector[start:end], cutSignal[start:end], ASKBinary[start:end], ASKBinary[start:end], "test2")	
	#graficar("ASK","ASK","amplitud","tiempo",newsignal[500:600])

processFile('handel.wav')
# Lo que viene a continuacion lo use para probar con que numero de bitspersec (que deberia 
# tener otro nombre porque no se que es en verdad) queda mas bonita la señal portadora
"""
for bits in range(1,150):
	time = np.linspace(0, 1, bits)
	C0 = 1 * np.cos(2 * np.pi * 5310 * time)
	C1 = 5 * np.cos(2 * np.pi * 5310 * time)

	graficar("C0" + str(bits),"C0","amplitud","tiempo",C0,time)
	graficar("C1" + str(bits),"C1","amplitud","tiempo",C1,time)
"""	