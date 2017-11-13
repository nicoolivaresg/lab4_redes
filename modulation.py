import numpy as np
from scipy.io.wavfile import read,write
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import resample
from scipy.integrate import cumtrapz

DPI = 100
FIGURE_WIDTH = 8
FIGURE_HEIGHT = 3

GRAPH_DIR = "graph/" 
AUDIO_DIR = "audio/"

TIMEXLABEL = "time"
FREQXLABEL = "frequency"
AMPLITUDEYLABEL = "amplitude"


AWGN_TITLE = "Additive White Gaussian Noise"
AM_TITLE = "Amplitude Modulation"
FM_TITLE = "Frequency Modulation"
AUDIO_NAME = "handel"
GRAPH_AWGN_NO = 0


# Funcion que usa el read de scipy.io.wavfile para abrir un archivo .wav y obtener la
# frecuencia y la informacion del sonido, esta funcion ademas obtiene un vector de tiempo
# dependiendo de la canidad de datos y la frecuencia del audio.
# Entrada:
# 	filename	- Nombre del archivo por abrir.
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
# Entrada:
#	filename	- Nombre del archivo .wav a crear (Ejemplo "salida.wav").
#	frequency	- Frecuencia de muestreo del audio.
#	signal		- La señal en el dominio del tiempo.
def save_wav_audio(filename, frequency, fsignal):
	write(AUDIO_DIR + filename + ".wav", frequency, fsignal.astype('int16'))


# Funcion que grafica los datos en ydata y xdata, y escribe los nombres del eje x, eje y,
# y el titulo de una figura. Esta figura la guarda en un archivo con el nombre filename.png.
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



def interpolate(signal, oldFreq, newFreq):
	nOld = len(signal)
	tOld = nOld / oldFreq; # Intervalo de tiempo
	tiemposOld = np.linspace(0, tOld, nOld)
	nNew = int(len(signal)*newFreq/oldFreq)
	tNew = nNew / newFreq; # Intervalo de tiempo
	tiemposNew = np.linspace(0, tNew, nNew)
	return (np.interp(tiemposNew,tiemposOld,signal),tiemposNew)


def AmplitudeModulation(modulatorSignal, carrierFreq, interpFreq, modulationpercentage):
	n = len(modulatorSignal)
	t = n / (4*carrierFreq); # Intervalo de tiempo
	carrier = generateCarrier(carrierFreq, interpFreq, t)
	return modulationpercentage * modulatorSignal * carrier

def FrequencyModulation(modulatorSignal,carrierFreq, interpFreq, modulationpercentage):
	n = len(modulatorSignal)
	time = n / interpFreq
	timeVector = np.linspace(0, time, n)
	signalIntegral = cumtrapz(modulatorSignal, timeVector, initial=0)
	w = 2 * np.pi * carrierFreq * timeVector
	modulation = np.cos(w + modulationpercentage * signalIntegral)
	return modulation

def fmMod(modulatorSignal,carrierFreq, interpFreq, modulationpercentage = 0.0):
	#Frecuencia de la portadora mas la de la señal
	freq = carrierFreq + modulatorSignal
	#Calculo de la frecuencia instantanea: 
	#integral instantanea: x(t) * dt
	#
	ifreq = np.cumsum(freq) * (1/interpFreq)
	modulacion = np.cos(2 * np.pi * ifreq)
	return modulacion


def generateCarrier(freq, interpFreq, duration):
	time = np.linspace(0,duration,duration*interpFreq)
	amplitude = np.cos(2 * np.pi * freq * time)
	return amplitude

def amMod(data,interpFreq, carrierFreq):
	#Tiempo de duracion de la señal
	time = len(data)/interpFreq
	#Generación de una portadora
	portadora = generateCarrier(carrierFreq,interpFreq,time)
	#MOdulación: data * portadora
	modulacion = data * portadora
	return modulacion

##### TESTING #####
def processFile(path):
	#Lectura de audios
	samplefreq1, data1, tiempos1 = load_wav_audio(path)
	carrierFreq = 5310
	interpFreq = 4 * carrierFreq

	nNew = int(len(data1)*interpFreq/samplefreq1)
	tNew = nNew / interpFreq; # Intervalo de tiempo
	tiemposNew = np.linspace(0, tNew, nNew)

	#Se interpola la señal original
	interpolatedSignal, tiemposInterpolated = interpolate(data1, samplefreq1, interpFreq)
	#Variables para suma
	samples = len(interpolatedSignal)
	zoom_percentage = 0.6
	zoom_in_start = 0
	zoom_in_stop = int(samples*zoom_percentage)
	
	
	#portadora = generateCarrier(carrierFreq,samplefreq1,len(interpolatedSignal)/samplefreq1)

	#graficar("cos_carrier", "Cos carrier", AMPLITUDEYLABEL, TIMEXLABEL, portadora[zoom_in_start:zoom_in_stop], np.linspace(0,samplefreq1,len(interpolatedSignal)/samplefreq1*samplefreq1)[zoom_in_start:zoom_in_stop])

	#resAM = amMod(interpolatedSignal, samplefreq1,carrierFreq)
	#graficar("am", "am", AMPLITUDEYLABEL, TIMEXLABEL, resAM[zoom_in_start:zoom_in_stop],  np.linspace(0,samplefreq1,len(interpolatedSignal)/samplefreq1*samplefreq1)[zoom_in_start:zoom_in_stop], color = 'r')

	resFM = fmMod(interpolatedSignal, samplefreq1,carrierFreq)
	graficar("fm", "fm", AMPLITUDEYLABEL, TIMEXLABEL, resFM[zoom_in_start:zoom_in_stop],  np.linspace(0,samplefreq1,len(interpolatedSignal)/samplefreq1*samplefreq1)[zoom_in_start:zoom_in_stop], color = 'r')

	#Se aplican la modulacion AM a la señal original
	amResults = [AmplitudeModulation(interpolatedSignal, carrierFreq, interpFreq, 0.15),AmplitudeModulation(interpolatedSignal, carrierFreq , interpFreq, 1.0),AmplitudeModulation(interpolatedSignal, carrierFreq , interpFreq, 1.25)]
	fmResults = [FrequencyModulation(interpolatedSignal, carrierFreq, interpFreq, 0.15),FrequencyModulation(interpolatedSignal, carrierFreq , interpFreq, 1.0),FrequencyModulation(interpolatedSignal, carrierFreq , interpFreq, 1.25)]

	#print(len(interpolatedSignal), samplefreq1)

	#Se aplica la FFT a la señal original
	#fftOriginalSignal, fftOriginalSignalSamples = fourier_transform(data1,samplefreq1)
	
	#graficar("original_fft", "Original Fourier Transform ", AMPLITUDEYLABEL, FREQXLABEL, abs(fftOriginalSignal), fftOriginalSignalSamples)
	#Se aplica la FFT a la señal original
	#fftOriginalSignal, fftOriginalSignalSamples = fourier_transform(interpolatedSignal,carrierFreq)
	#graficar("original_interpolated_fft", "Original Fourier Transform (interpolated)", AMPLITUDEYLABEL, FREQXLABEL, abs(fftOriginalSignal), fftOriginalSignalSamples)
	

	#fftam, fftsamplesam = fourier_transform(resAM,carrierFreq)
	
	#graficar("am_fft", "AM Fourier Transform ", AMPLITUDEYLABEL, FREQXLABEL, abs(fftam), fftsamplesam)

	fftfm, fftsamplesfm = fourier_transform(resFM,carrierFreq)
	
	graficar("fm_fft", "FM Fourier Transform ", AMPLITUDEYLABEL, FREQXLABEL, abs(fftfm), fftsamplesfm)
	#Se aplica la transformada de Fourier a cada modulación y se grafica
	for i in range(0,len(amResults)):
		fftAMSignal, fftAMSignalSamples = fourier_transform(amResults[i],interpFreq)
		graficar("AMfft"+str(i), "AM Fourier Transform " + str(i), AMPLITUDEYLABEL, FREQXLABEL, abs(fftAMSignal), fftAMSignalSamples)

	for i in range(0,len(fmResults)):
		fftFMSignal, fftFMSignalSamples = fourier_transform(fmResults[i],interpFreq)
		graficar("FMfft"+str(i), "FM Fourier Transform " + str(i), AMPLITUDEYLABEL, FREQXLABEL, abs(fftFMSignal), fftFMSignalSamples)

	#Gráficas variadas
	
	#Señal original
	modulation_percentage = [15,100,125]

	#graficar(AUDIO_NAME  , "Original signal: " + AUDIO_NAME  , AMPLITUDEYLABEL , TIMEXLABEL , interpolatedSignal)
	for i in range(0,len(modulation_percentage)):
		graficar("AM-" + AUDIO_NAME + str(modulation_percentage[i]), AM_TITLE + " " + str(modulation_percentage[i]), AMPLITUDEYLABEL, TIMEXLABEL, amResults[i] , tiemposNew)	
	
	for i in range(0,len(modulation_percentage)):
		graficar("FM-" + AUDIO_NAME + str(modulation_percentage[i]), FM_TITLE + " " + str(modulation_percentage[i]), AMPLITUDEYLABEL, TIMEXLABEL, amResults[i] , tiemposNew)	
	


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