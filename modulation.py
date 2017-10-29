import numpy as np
import math
from scipy.io.wavfile import read,write
import matplotlib.pyplot as plt

DPI = 100
FIGURE_WIDTH = 8
FIGURE_HEIGHT = 3

GRAPH_DIR = "graph/" 
AUDIO_DIR = "audio/"

TIMEXLABEL = "time"
AMPLITUDEYLABEL = "amplitude"

AWGN_TITLE = "Additive White Gaussian Noise"
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



def AmplitudeModulation(moduladorasignal, frecuenciaportadora):
	modulada = moduladorasignal*math.cos(frecuenciaportadora)
	return modulada




def generateAWGN(mean, stddev, length):
	gaussianoise = np.random.normal(mean, stddev, length)
	return gaussianoise

def applyAWGN(signal, samplefreq):
	awgn = generateAWGN(0,0.1, int(samplefreq/2))
	i=0
	while i<int(samplefreq/2):
		signal[i] += awgn[i]
		i+=1
	return signal


##### TESTING #####
def processFile(path):
	#Lectura de audios
	samplefreq1, data1, tiempos1 = load_wav_audio(path)


	#Aplicacion de Additive White Gaussian Noise
	AWGNaplication = applyAWGN(data1,samplefreq1)

	#Ruido aplicado
	ruidoAWGN = generateAWGN(0,0.1,int(samplefreq1/2))

	#Gráficas variadas
	
	#Señal original
	graficar(AUDIO_NAME  , "Original signal: " + AUDIO_NAME  , AMPLITUDEYLABEL , TIMEXLABEL , data1)
	
	#AWGN
	graficar("applied_noise","AWGN distribución N(0;0,1)", AMPLITUDEYLABEL, TIMEXLABEL, ruidoAWGN)

	#Señal con AWGN aplicadas
	graficar("AWGN_"+AUDIO_NAME , "Applied AWGN to " + AUDIO_NAME , AMPLITUDEYLABEL , TIMEXLABEL , AWGNaplication)

	#Salidas de audio variadas
	
	#Señal portadora
	save_wav_audio("AWGN_"+AUDIO_NAME ,samplefreq1, AWGNaplication)
	

processFile("handel.wav")