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

##########
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
##########
def load_wav_audio(filename):
	frecuencia, datos = read(filename)
	n = len(datos)
	Ts = n / frecuencia; # Intervalo de tiempo
	tiempos = np.linspace(0, Ts, n) # Tiempo en segundos para cada dato de 'datos'
	return (frecuencia, datos, tiempos)


##########
# Proceso que crea una figura que tiene 3 graficos haciendo uso de la funcion subplot()
# que trae matplotlib.
# 
# Entrada:
#	XVector		- Vector comun x para los 3 graficos.
# 	YVector1 	- Vector de valores y para el grafico de más arriba.
# 	YVector2 	- Vector de valores y para el grafico del medio.
#	YVector3 	- Vector de valores y para el grafico de mas abajo.
#	title1		- Titulo del primer grafico.
#	title2		- Titulo del segundo grafico.
#	title3		- Titulo del tercero grafico.
#	filename	- Nombre del archivo en el que se guarda la figura (sin extension).
##########
def triple_subplot(XVector, YVector1, YVector2, YVector3, title1, title2, title3, filename):
	plt.subplots_adjust(hspace=.5)
	plt.subplot(311)
	plt.title(title1, fontsize=12)
	plt.plot(XVector, YVector1, linewidth=0.4)
	plt.ylabel("Amplitud")

	plt.subplot(312)
	plt.title(title2, fontsize=10)
	plt.plot(XVector, YVector2, linewidth=0.4)
	plt.ylabel("Amplitud")

	plt.subplot(313)
	plt.title(title3, fontsize=10)
	plt.xlabel("Tiempo[s]")
	plt.ylabel("Amplitud")
	plt.plot(XVector, YVector3, linewidth=0.4)
	plt.savefig(GRAPH_DIR + filename + ".png", bbox_inches='tight')
	plt.clf()
	plt.close('all')

##########
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
##########
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


##########
# Funcion que realiza un filtro pasabajo para una señal. Se usa para terminar el proceso de
# demodulamiento de una señal modulada en amplitud (AM).
# 
# Entrada:
#	fsignal		- Señal a la que se le realiza el filtro pasabajo.
# 	sampleFreq 	- Frecuencia de muestreo de la señal (frecuencia de interpolacion)
# 	targetFreq 	- Frecuencia a la que se quiere realiza el filtro (highcut)
#					Es la frecuencia de muestreo de la señal original.
#
# Salida:
#	z	- Señal filtrada con un filtro pasabajo en la frecuencia targetFreq / 2.
##########
def lowpass(fsignal, sampleFreq, targetFreq):
	nyq = 0.5 * sampleFreq
	high = (targetFreq / 2) / nyq
	if(high > 1):
		high = 0.99999
	b, a = signal.butter(9, high, 'low')
	z = signal.filtfilt(b, a, fsignal)
	return z


##########
# Funcion que realiza la modulacion ASK (Amplitude Shift Keying) de una señal
# digitalizada, retorna la señal modulada de acuerdo a los parametros entregados
# para las señales portadoras que se utilzan.
#
# Entrada:
#	bitspersec	- Cuantos valores de la portadora hay en "1 segundo"
#	A 			- Amplitud de la portadora que simboliza el bit 0
#	B 			- Amplitud de la portadora que simboliza el bit 1 (debe ser mayor que A)
#	carrierFreq - Frecuencia de la señal portadora
#	signal 		- Señal previamente digitalizada que se quiere modular.
#
# Salida:
#	modulatedSignal - Arreglo numpy con la señal digital modulada por el metodo ASK.
##########
def ASK_modulation(bitspersec, A, B, carrierFreq, signal):
	# Vector de tiempo
	timeVector = np.linspace(0, 1, bitspersec)
	if carrierFreq != 5310:
		# Formula del profe
		timeVector = np.arange(0, 5/carrierFreq, 1/(carrierFreq*10.5))
	# Portadora para 1
	C1 = B * np.cos(2 * np.pi * carrierFreq * timeVector)
	# Portadora para 0
	C0 = A * np.cos(2 * np.pi * carrierFreq * timeVector)

	# Arreglo para almacenar resultados de los bits reconocidos de la señal
	y = []
	for bit in signal:
		if bit:
			y.extend(C1)
		else:
			y.extend(C0)
	return np.array(y)


##########
# Realiza la demodulacion de una señal modulada por ASK utilizando las señales
# portadors coseno y seno. Finalmente retorna la señal digital demodulada.
# digital.
# 
# Entrada:
#	B 	 		- Amplitud que fue utilizada para simbolizar el bit 1.
#	carrierFreq - Frecuencia de las señales portadoras a utilizar.
#	signal 		- Señal modulada por el metodo ASK.
#	ratio		- Cantidad de elementos que tiene la señal modulada por sobre
#					la señal digital previa interpolaciones.
#
# Salida:
#	demodulatedSignal - Arreglo numpy con la señal digital demodulada.
##########
def ASK_demodulation(B, carrierFreq, signal, ratio):
	ctimeVector = np.linspace(0, 80.5/carrierFreq, len(signal))
	cosCarrier = B * np.cos(2 * np.pi * carrierFreq * ctimeVector)
	sinCarrier = B * np.sin(2 * np.pi * carrierFreq * ctimeVector)

	productCos = abs(signal * cosCarrier)
	productSin = abs(signal * sinCarrier)

	product = []
	for i in range(0, len(productCos)):
		product.append(max(productCos[i], productSin[i]))

	product = np.array(product)
	filteredSignal = lowpass(product, 8192, carrierFreq/12)

	extendedDemodulatedSiganl = []
	for amplitude in filteredSignal:
		if amplitude > B ** 1.5:
			extendedDemodulatedSiganl.append(1)
		else:
			extendedDemodulatedSiganl.append(0)

	demodulatedSignal = []
	for i in range(16, len(extendedDemodulatedSiganl), int(ratio)):
		demodulatedSignal.append(extendedDemodulatedSiganl[i])

	return np.array(demodulatedSignal)


##########
# Realiza la digitalizacion de una señal, transformando cada una de las
# amplitudes de la señal analoga en un numero binario de largo 16.
# 
# Entrada:
#	x 	- Señal que se quiere digitalizar.
#
# Salida:
#	digitalSignal - Arreglo numpy de bits, es la señal digitalizada.
##########
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


##########
# Recibe una señal, le agrega ruido blanco aditivo gaussiano normal, para
# luego retorna la señal con ruido.
# Esta funcion recibe un rango de SNR en decibeles, para luego obtener datos
# aleatorios para cada uno de ellos y crear el ruido que se le agrega a la señal.
# 
# Entrada:
#	signal 			- Señal a la que se le aplica el ruido.
#	snrdbLowerLimit	- Signal-to-noise-ratio minimo en decibeles.
#	snrdbUpperLimit	- Signal-to-noise-ratio maximo en decibeles.
#
# Salida:
#	noisySignal - Señal con ruido agregado.
##########
def awgn(signal, snrdbLowerLimit, snrdbUpperLimit):
	avgEnergy = sum(abs(signal)* abs(signal)) / len(signal)
	snrdbRandom = np.random.uniform(snrdbLowerLimit, snrdbUpperLimit, len(signal))
	snrLinear = calculateSnrLinear(snrdbRandom)
	noiseVariance = avgEnergy / (2 * snrLinear)

	noise = np.sqrt(2* noiseVariance) * np.random.randn(len(signal))

	return signal + noise


##########
# Obtiene el signal-to-noise-ratio lineal de acuerdo al snr en decibeles.
# 
# Entrada:
#	snrdb - Signal-to-noise-ratio en decibeles.
#
# Salida:
#	snrLineal - Signal-to-noise-ratio lineal.
##########
def calculateSnrLinear(snrdb):
	return 10 ** (snrdb/10.0)


##########
# Procesa el audio que se encuentra en path, para realizar los siguientes procesos:
#	- Digitalizacion de la señal
#	- Modulacion y demodulacion de la señal digital
#	- Agregarle ruido AWGN a la señal modulada para luego modularla.
#	- Calcular la tasa de error entre la señal digital y la señal demodulada con ruido.
# 
# Entrada:
#	path - String con el camino dondel se encuentra el archivo de audio que se quiere procesar.
#	A 	 - Amplitud a utilizar en la modulacion ASK para representar el bit 0
#	B 	 - Amplitud a utilizar en la modulacion ASK para representar el bit 1
#	savePlot - Indica si es que se deben guardar los graficos.
# Salida:
#	errorRate - Porcentaje de error entre la señal demodulada con error y la señal digital.
##########
def processFile(path, A, B, savePlot=False):
	carrierFreq = 5310
	bitspersec = 33
	dataToUse = 800 # Datos que se usan de la señal original
	start = 600	# Desde que dato
	end = 2600 # Hasta que dato se grafica
	samplingRate, signal, timeSignalVector = load_wav_audio(path)
	signalTime = len(signal)/samplingRate

	# Se trabaja con una muestra de la señal para hacerla mas manejable/computable
	signalSample = signal[:dataToUse]
	signalSampleTime = signalTime * (len(signalSample) / len(signal))

	#Preproceso de señal, conversion a bits
	digitalSignal = preProcessSignal(signalSample)

	# Modulacion de la señal digital
	ASKSignal = ASK_modulation(bitspersec, A, B, carrierFreq, digitalSignal)
	noisyASKSignal = awgn(ASKSignal, -2, 10)

	# Datos usados para interpolar la señal digitalizada para que tenga la misma cantidad
	# de datos que la modulacion y se puedan graficar con el mismo vector de tiempo
	oldX = np.linspace(0, signalSampleTime, len(digitalSignal))
	newX = np.linspace(0, signalSampleTime, len(ASKSignal))
	digitalSignalI = np.interp(newX, oldX, digitalSignal)

	# Vector de tiempo ajustado a la muestra de la señal original y al largo de la señal 
	# digital interpolada.
	timeVector = np.linspace(0, signalSampleTime, len(digitalSignalI))
	
	# Demodulacion de las señales (con y sin ruido) e interpolacion para poder graficarlas
	# en el mismo tiempo que tiene la modulacion ASK
	demodulatedSignal = ASK_demodulation(B, carrierFreq, ASKSignal, len(newX) / len(oldX))
	demodulatedSignalI = np.interp(newX, oldX, demodulatedSignal)

	demodulatedNoisySignal = ASK_demodulation(B, carrierFreq, noisyASKSignal, len(newX) / len(oldX))
	demodulatedNoisySignalI = np.interp(newX, oldX, demodulatedNoisySignal)

	# Calcular la tasa de error
	errors = (demodulatedNoisySignal != digitalSignal).sum()
	errorRate = 100.0 * errors / len(demodulatedNoisySignal)
	print("Tasa de errores con amplitud B de " + str(B) + ": " + str(errorRate) + "%")

	# Graficar la señal digital, su modulacion, la modulacion con ruido y las demodulaciones correspondientes
	if savePlot == True:
		triple_subplot(timeVector[start:end], digitalSignalI[start:end], ASKSignal[start:end], demodulatedSignalI[start:end],
						"Señal digital", "Señal modulada por ASK", "Señal demodulada sin ruido", "ASK_dig_noiseless" + str(B))
		triple_subplot(timeVector[start:end], digitalSignalI[start:end], noisyASKSignal[start:end], demodulatedNoisySignalI[start:end],
						"Señal digital", "Señal modulada por ASK + ruido AWGN", "Señal demodulada con ruido", "ASK_dig_noisy" + str(B))
	return errorRate

# Se ejecuta el programa con distintas amplitudes para los bits 1 para mostrar
# el efecto del ruido al tener amplitudes mas similares en las señales portadoras
# (modulacion Amplitude Shift Keying (ASK))
processFile('handel.wav', 1, 5, True)
processFile('handel.wav', 1, 8, True)

lowerLimitB = 3
upperLimitB = 12
Belements = 10 * (upperLimitB - lowerLimitB) + 1
B = np.linspace(lowerLimitB, upperLimitB, Belements)
errorRates = []
print(B)
for b in B:
	errorRates.append(processFile('handel.wav', 1, b))

errorRates = np.array(errorRates)

graficar("error_rate", "Tasa de error de acuerdo a la amplitud del bit 1", "Tasa de error [%]", "Amplitud de la señal portadora que representa el bit 1", errorRates, B)



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