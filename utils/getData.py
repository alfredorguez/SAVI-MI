import numpy as np

def get_data():
    from scipy.io import loadmat
    # Obtención de datos en crudo
    raw_data = loadmat("datos/dataicann.mat")
    muestras = raw_data['z']
    etiquetas = raw_data['nom']
    nombresCaracteristicas = raw_data['NombresVariables']

    # Concatenar todas las muestras en una matriz bidimensional
    X = np.vstack([np.vstack(muestra) for muestra in muestras[0]])

    # Crear un vector de etiquetas con el mismo número de filas que muestras_concatenadas
    etiq_str = np.array([str(etiquetas[0][k][0]) for k in range(etiquetas.shape[1])])
    etiq_str = np.array([etiqueta.split('.')[0] for etiqueta in etiq_str])
    Y = np.repeat(etiq_str, [muestra.shape[0] for muestra in muestras[0]])

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # Ajustar y transformar el vector de etiquetas
    Y_int = le.fit_transform(Y)

    # Crear un objeto StandardScaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # Ajustar y transformar la matriz X
    Xn = scaler.fit_transform(X)

    fm = 5000 # Frecuencia de muestreo
    tm = 1/fm # Tiempo de muestreo
    Q = X.shape[0] # Número de muestras
    ff = np.arange(0, Q, 1) * (fm/Q) # Vector de frecuencias

    # Parámetros constructivos de las ventanas
    N = 1000 # Tamaño de la ventana
    delta = 200 # Desplazamiento de la ventana
    bandasFreq = np.arange(25, 525, 25) # Bandas de frecuencia

    f = np.arange(0, N, 1) * (fm/N) # Vector de frecuencias

    # Inicialización de matriz de características
    F = np.zeros(len(bandasFreq) * Xn.shape[1])
    Fy = np.array([0])
    Fx = []

    # Método de enventanado
    for k in range(0, Q-N, delta):
        # Obtención de arreglo con índices de la ventana
        idx = np.arange(k, k+N, 1)
        
        # Inicialización de vector de características
        carVector = np.zeros(len(bandasFreq) * Xn.shape[1]) # Número de bandas de frecuencia * número de características

        for i, f_banda in enumerate(bandasFreq):
            for j in range(Xn.shape[1]):
                # Seleccionar la ventana de la característica j
                v = Xn[idx, j]
                # Calcular la FFT
                V = np.abs(np.fft.fft(v))
                # Seleccionar los armónicos en la banda de frecuencia deseada
                armonicosSeleccionados = V[(f >= f_banda - 5) & (f < f_banda + 5)]
                # Calcular el valor eficaz (RMS) y almacenarlo en el vector de características
                carVector[i * Xn.shape[1] + j] = np.sqrt(np.sum(armonicosSeleccionados**2) / len(armonicosSeleccionados))
                
        # Cálculo de la etiqueta de la ventana
        vy = int(round(Y_int[idx].mean()))
        
        # Añadir vector de características a la matriz
        F = np.vstack((F, carVector))
        Fy = np.append(Fy, vy)

    # Crear lista de etiquetas de características
    for f_banda in bandasFreq:
        for k in range(Xn.shape[1]):
            Fx.append(str(nombresCaracteristicas[0][k][0])+'@'+str(f_banda)+'Hz')

    # Eliminar primera fila de la matriz de características
    F = F[1:,:]
    Fy = Fy[1:]

    # Transformación de etiquetas de muestras de enteros a strings
    Fy = le.inverse_transform(Fy).tolist()

    return F, Fy, Fx

if __name__ == "__main__":
    F, Fy, Fx = get_data()
    print(F.shape)
    print(Fy.shape)
    print(Fx)
    print(len(Fx))