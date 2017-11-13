import numpy as np
from scipy.io.wavfile import read,write
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.integrate import cumtrapz

DPI = 100
FIGURE_WIDTH = 8
FIGURE_HEIGHT = 3

GRAPH_DIR = "graph/" 
AUDIO_DIR = "audio/"

TIMEXLABEL = "Tiempo [s]"
FREQXLABEL = "Frecuencia [Hz]"
AMPLITUDEYLABEL = "Amplitud"


AWGN_TITLE = "Additive White Gaussian Noise"
AM_TITLE = "Amplitude Modulation"
FM_TITLE = "Frequency Modulation"
AUDIO_NAME = "handel"
GRAPH_AWGN_NO = 0


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


# Funcion que usa el write de scipy.io.wavfile para escribir un archivo .wav de acuerdo a 
# los parametros entregados a esta funcion.
#
# Entrada:
#	filename	- Nombre del archivo .wav a crear (Ejemplo "salida.wav").
#	frequency	- Frecuencia de muestreo del audio.
#	signal		- La señal en el dominio del tiempo.
def save_wav_audio(filename, frequency, fsignal):
	write(AUDIO_DIR + filename + ".wav", frequency, fsignal.astype('int16'))


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
	plt.xlabel("Tiempo [s]")
	plt.ylabel("Amplitud")
	plt.plot(XVector, YVector3, linewidth=0.4)
	plt.savefig(GRAPH_DIR + tipo + "-" + str(percentage) + ".png", bbox_inches='tight')
	plt.clf()
	plt.close('all')


# Funcion que hace uso de la tranformada de fourier, fft() y fftfreq(), para obtener la
# secuencia de valores de los datos obtenidos del audio y para obtener las frecuencias
# de muestreo (que depende de la frecuencia del audio y del largo del audio) respectivamente.
# Entrada:
#	data		- Datos obtenidos al leer el archivo de audio con scipy o con load_wav_audio().
#	frequency	- Numero entero con la frecuencia de muestreo del audio en [Hz].
# Salida:
#	fftValues	- Transformada de fourier normalizada para los valores en data.
#	fftSamples	- Frecuencias de muestreo que dependen del largo del arreglo data y la frequency.
def fourier_transform(data, frequency):
	n = len(data)
	Ts = n / frequency
	fftValues = fft(data) / n # Computacion y normalizacion
	fftSamples = np.fft.fftfreq(n, 1/frequency)

	return (fftValues, fftSamples)


# Transforma una frecuencia (targetFreq) a su equivalente en frecuencia Nyquist, de acuerdo 
# a la frecuencia de muestreo de la señal, esta frecuencia es por defecto 44100.
# Entrada:
#	targetFreq		- Frecuencia de la que se quiere obtener su equivalente en Nyquist frequency.
#	sampleRate		- Frecuencia de muestreo de la señal (por defecto es 44100).
# Salida:
# 	nyq_frequency	- Frecuencia transformada a su equivalente en frecuencia Nyquist.
def get_nyq(targetFreq, sampleRate=8192):
	if(targetFreq > sampleRate):
		return 1
	else:
		return np.divide(targetFreq, sampleRate / 2.0)


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
def lowpass(fsignal, sampleFreq, targetFreq):
	nyq = 0.5 * sampleFreq
	high = (targetFreq / 2) / nyq
	if(high > 1):
		high = 0.99999
	b, a = signal.butter(9, high, 'low')
	z = signal.filtfilt(b, a, fsignal)
	return z


# Funcion que se usa para realizar la interpolacion de una señal de audio.
# Esta funcion recibe una señal, su frecuencia y la frecuencia a la que se quiere llegar,
# luego usando estos valores rellena con los puntos que le faltan a la señal para estar
# muestreada al mismo periodo que una señal de la frecuencia a la que se quiere llegar.
# 
# Entrada:
#	signal	- Señal que se quiere interpolar.
# 	oldFreq - Frecuencia de la señal que se quiere interpolar.
# 	newFreq - Frecuencia en la que se quiere interpolar (frecuencia de destino).
#
# Salida:
#	interpolatedSignal - Señal interpolada en la nueva frecuencia.
def interpolate(signal, oldFreq, newFreq):
	nOld = len(signal)
	tOld = nOld / oldFreq; # Intervalo de tiempo
	tiemposOld = np.linspace(0, tOld, nOld)
	nNew = int(len(signal)*newFreq/oldFreq)
	tNew = nNew / newFreq; # Intervalo de tiempo
	tiemposNew = np.linspace(0, tNew, nNew)
	return (np.interp(tiemposNew,tiemposOld,signal),tiemposNew)


# Modulacion en amplitud de una señal (AM).
# 
# Entrada:
#	modulatorSignal		 - Señal que se quiere modular en amplitud.
# 	carrierFreq 		 - Frecuencia en la que se quiere mover la señal original.
#							En otras palabras, la frecuencia de la señal portadora.
# 	interpFreq 			 - Frecuencia de la señal interpolada que se quiere modular.
#	modulationPercentage - Porcentaje de modulacion que se quiere utilizar.
#
# Salida:
#	modulatedSignal - Señal modulada en amplitud (AM).
def AmplitudeModulation(modulatorSignal, carrierFreq, interpFreq, modulationPercentage):
	n = len(modulatorSignal)
	t = n / interpFreq; # Intervalo de tiempo
	carrier = generateCarrier(carrierFreq, interpFreq, t, n)
	return modulationPercentage * modulatorSignal * carrier

# Demodulacion de una señal modulada en amplitud.
# 
# Entrada:
#	modulatedSignal	- Señal modulada en amplitud (AM).
# 	carrierFreq 	- Frecuencia de la portadora que se utilizó para modular la señal.
# 	interpFreq 		- Frecuencia de la señal interpolada que se quiere modular.
#	sampleFreq		- Frecuencia de muestreo de la señal original.
#
# Salida:
#	signal - Señal demodulada, se puede grabar en audio.
def AmplitudDemod(modulatedSignal, carrierFreq, interpFreq, sampleFreq):
	n = len(modulatedSignal)
	t = n / interpFreq # Intervalo de tiempo
	carrier = generateCarrier(carrierFreq, interpFreq, t, n)
	signal = modulatedSignal * carrier
	newSignal = lowpass(signal, interpFreq, sampleFreq)
	return newSignal

# Modulacion en frecuencia de una señal (FM).
# 
# Entrada:
#	modulatorSignal		 - Señal que se quiere modular en frecuencia (FM).
# 	carrierFreq 		 - Frecuencia de la señal portadora.
# 	interpFreq 			 - Frecuencia de la señal interpolada que se quiere modular.
#	modulationPercentage - Porcentaje de modulacion que se quiere utilizar.
#
# Salida:
#	modulatedSignal - Señal modulada en frecuencia (FM).
def FrequencyModulation(modulatorSignal,carrierFreq, interpFreq, modulationPercentage):
	n = len(modulatorSignal)
	time = n / interpFreq
	timeVector = np.linspace(0, time, n)
	signalIntegral = cumtrapz(modulatorSignal, timeVector, initial=0)
	w = 2 * np.pi * carrierFreq * timeVector
	modulation = np.cos(w + modulationPercentage * signalIntegral)
	return modulation

# Demodulacion de una señal modulada en frecuencia (FM).
# Puede tener problemas si es que es muy poco el ancho de banda utilizado.
# 
# Entrada:
#	modulatedSignal	- Señal modulada en frecuencia (FM).
# 	carrierFreq 	- Señal de la portadora utilizada en la modulacion.
# 	timeVector 		- Vector de tiempo para los datos de la señal.
#
# Salida:
#	signal - Señal demodulada, se puede grabar en audio.
def FrequencyDemod(modulatedSignal, carrierFreq, timeVector):
	tmp = signal.hilbert(modulatedSignal)
	baseband = tmp * np.exp(-2 * np.pi * carrierFreq * timeVector * 1j)
	tmp = baseband[1::1] * np.conjugate(baseband[0:-1:1])
	return np.angle(baseband)

# Genera una señal portadora usando el coseno.
# 
# Entrada:
#	freq		- Frecuencia de la señal a crear.
# 	interpFreq 	- Frecuencia de la señal interpolada.
#					Normalmente es 4 veces el valor de la frecuencia de la portadora.
# 	duration	- Tiempo que dura la señal con la que se quiere cruzar la portadora.
#	n 			- Cantidad de datos en la señal interpolada.
#
# Salida:
#	carrier - Señal portadora con frecuencia freq.
def generateCarrier(freq, interpFreq, duration, n = 0):
	if n == 0:
		n = duration*interpFreq
	time = np.linspace(0,duration,n)
	amplitude = np.cos(2 * np.pi * freq * time)
	return amplitude


##### TESTING #####
def processFile(path):
	#Lectura de audios
	samplefreq1, data1, tiempos1 = load_wav_audio(path)
	carrierFreq = 11025
	interpFreq = 4 * carrierFreq

	graficar(AUDIO_NAME, "Señal original", AMPLITUDEYLABEL, TIMEXLABEL, data1, tiempos1)

	#Se interpola la señal original
	interpolatedSignal, tiemposInterpolated = interpolate(data1, samplefreq1, interpFreq)
	# Variables para realizar el zoom
	samples = len(interpolatedSignal)
	zoom_percentage = 0.001
	zoom_in_start = 1000
	zoom_in_stop = zoom_in_start + int(samples*zoom_percentage)
 
 	# Crear un vector de tiempo para la señal interpolada (aunque ya la trae el interpolate)
	n = len(interpolatedSignal)
	time = n / interpFreq
	timeVector = np.linspace(0, time, n)

	# Arreglo con los porcentajes a utilizar en las modulaciones
	modulation_percentage = [15,100,125,475]
	modulations = len(modulation_percentage)
	amResults = [None] * modulations
	fmResults = [None] * modulations
	#Se aplican la modulacion AM a la señal original
	for i in range(0,len(modulation_percentage)):
		AMresult = AmplitudeModulation(interpolatedSignal, carrierFreq, interpFreq, modulation_percentage[i] / 100.0)
		amResults[i] = AMresult

		# Transformada de Fourier de la señal modulada y el grafico
		fftAMSignal, fftAMSignalSamples = fourier_transform(AMresult,interpFreq)
		graficar("AM-fft-"+str(modulation_percentage[i]), "AM Fourier Transform " + str(modulation_percentage[i])+"%", AMPLITUDEYLABEL, FREQXLABEL, abs(fftAMSignal), fftAMSignalSamples)

		# Demodular la señal, guardar su audio y graficar su transformada
		amDemod = AmplitudDemod(AMresult, carrierFreq, interpFreq, samplefreq1)
		save_wav_audio(AUDIO_NAME+"-AM-"+str(modulation_percentage[i])+"-demod", interpFreq, amDemod)
		fftAMSignal, fftAMSignalSamples = fourier_transform(amDemod, samplefreq1)
		graficar("demod-AM-fft-"+str(modulation_percentage[i]), "AM Demodulated Fourier Transform " + str(modulation_percentage[i])+"%", AMPLITUDEYLABEL, FREQXLABEL, abs(fftAMSignal), fftAMSignalSamples)
	
	#Se aplican la modulacion FM a la señal original
	for i in range(0,len(modulation_percentage)):
		FMresult = FrequencyModulation(interpolatedSignal, carrierFreq, interpFreq, modulation_percentage[i] / 100.0)
		fmResults[i] = FMresult

		# Obtener y graficar la transformada de Fourier de la señal modulada en FM
		fftFMSignal, fftFMSignalSamples = fourier_transform(fmResults[i],interpFreq)
		graficar("FM-fft"+str(modulation_percentage[i]), "FM Fourier Transform " + str(modulation_percentage[i])+"%", AMPLITUDEYLABEL, FREQXLABEL, abs(fftFMSignal), fftFMSignalSamples)

		# Demodular la señal, guardar su audio y graficar su transformada
		fmDemod = FrequencyDemod(FMresult, carrierFreq, timeVector) * samplefreq1
		save_wav_audio(AUDIO_NAME+"-FM-"+str(modulation_percentage[i])+"-demod", interpFreq, fmDemod)
		fftFMSignal, fftFMSignalSamples = fourier_transform(fmDemod,interpFreq)
		graficar("demod-FM-fft-"+str(modulation_percentage[i]), "FM Demodulated Fourier Transform " + str(modulation_percentage[i])+"%", AMPLITUDEYLABEL, FREQXLABEL, abs(fftFMSignal), fftFMSignalSamples)

	#Graficos con zoom para las modulaciones AM
	for i in range(0,len(modulation_percentage)):
		graficar("AM-" + AUDIO_NAME + "-" + str(modulation_percentage[i]), AM_TITLE + " " + str(modulation_percentage[i])+"%", AMPLITUDEYLABEL, TIMEXLABEL, amResults[i][zoom_in_start:zoom_in_stop] , timeVector[zoom_in_start:zoom_in_stop])	
	
	#Graficos con zoom para las modulaciones FM
	for i in range(0,len(modulation_percentage)):
		graficar("FM-" + AUDIO_NAME + "-" + str(modulation_percentage[i]), FM_TITLE + " " + str(modulation_percentage[i])+"%", AMPLITUDEYLABEL, TIMEXLABEL, fmResults[i][zoom_in_start:zoom_in_stop] , timeVector[zoom_in_start:zoom_in_stop])	
	

	# Graficos triples
	n = len(interpolatedSignal)
	time = n / interpFreq
	timeVector = np.linspace(0, time, n)
	carrier = generateCarrier(carrierFreq, interpFreq, time, n)
	zoom_data = 500
	zoom_in_start = 1000
	zoom_in_stop = zoom_in_start + zoom_data
	# En AM
	for i in range(0,len(modulation_percentage)):
		triple_subplot(timeVector[zoom_in_start:zoom_in_stop], interpolatedSignal[zoom_in_start:zoom_in_stop], carrier[zoom_in_start:zoom_in_stop], amResults[i][zoom_in_start:zoom_in_stop], "AM", modulation_percentage[i])

	#Aplicacion de Additive White Gaussian Noise
	#AWGNaplication = applyAWGN(data1,samplefreq1)

	#Ruido aplicado
	#ruidoAWGN = generateAWGN(0,0.1,int(samplefreq1/2))
	
	#AWGN
	#graficar("applied_noise","AWGN distribución N(0;0,1)", AMPLITUDEYLABEL, TIMEXLABEL, ruidoAWGN)

	#Señal con AWGN aplicadas
	#graficar("AWGN_"+AUDIO_NAME , "Applied AWGN to " + AUDIO_NAME , AMPLITUDEYLABEL , TIMEXLABEL , AWGNaplication)

	#Salidas de audio variadas
	
	#Señal portadora
	
	

processFile("handel.wav")