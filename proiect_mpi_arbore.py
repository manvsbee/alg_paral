from mpi4py import MPI
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from sklearn.metrics import confusion_matrix

# Impartim datele de testare la procesele MPI
def imparte_date(date, numar_procese, rank):
    numar_date = len(date)
    marime_portie_date = numar_date // numar_procese
    inceput = rank * marime_portie_date
    sfarsit = (rank + 1) * marime_portie_date if rank < numar_procese - 1 else numar_date
    return date[inceput:sfarsit]

# Procesam datele prin reducerea pixelilor de la 0-255 la 0-1
def procesare_date(date):
    return date.astype('float32') / 255.0

# Modele cu 'Convolution Layer', 'Pooling Layer', 'Flatten' si 'Dropout'
def fun_model_v1():
    model_v1 = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model_v1

def fun_model_v2():
    model_v2 = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model_v2

def fun_model_v3():
    model_v3 = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.35),

        keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.5),

        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model_v3
    
# Invatam modelele
def invatare_model_v1(model_v1, imagini_invatare, etichete_invatare):
    model_v1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_v1.fit(imagini_invatare, etichete_invatare, epochs=10, batch_size=64)

def invatare_model_v2(model_v2, imagini_invatare, etichete_invatare):
    model_v2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_v2.fit(imagini_invatare, etichete_invatare, epochs=10, batch_size=64)

def invatare_model_v3(model_v3, imagini_invatare, etichete_invatare):
    model_v3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_v3.fit(imagini_invatare, etichete_invatare, epochs=10, batch_size=64)
    
# Reducem gradientii
def reducere_gradienti(model):
    gradienti = [gradient.numpy() for gradient in model.trainable_weights]
    gradienti_redusi = [np.zeros_like(gradient) for gradient in gradienti]
    for i in range(len(gradienti)):
        comm.Allreduce(gradienti[i], gradienti_redusi[i], op=MPI.SUM)
    gradienti = [tf.convert_to_tensor(gradient) for gradient in gradienti_redusi]
    model.optimizer.apply_gradients(zip(gradienti, model.trainable_weights))

def rank_pentru_noduri(rank, num_processes):
    nod_stanga = 2 * rank + 1
    nod_dreapta = 2 * rank + 2
    if nod_stanga >= num_processes:
        nod_stanga = MPI.PROC_NULL
    if nod_dreapta >= num_processes:
        nod_dreapta = MPI.PROC_NULL
    return nod_stanga, nod_dreapta

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    numar_procese = comm.Get_size()
    rank = comm.Get_rank()

    # Stocam preciziile intr-o singura variabila pentru afisare
    rezultate_precizii = []

    rank_nod_stanga, rank_nod_dreapta = rank_pentru_noduri(rank, numar_procese)

    # Bariera pentru sincronizare in printare
    print("Procesul %d are in nodul din stanga procesul: %d si in nodul din dreapta: %d" % (rank, rank_nod_stanga, rank_nod_dreapta))
    comm.Barrier()


    if rank == 0:
        print("Proiect Algoritmi Paraleli")
        print("Neural Nets")
        print("Topologie Arbore:")
        print("1. Precizie folosind modelul 1")
        print("2. Precizie folosind modelul 2")
        print("3. Precizie folosind modelul 3")
        print("0. Inchide programul")
        alegere = input("Raspuns: ")
        alegere = comm.bcast(alegere, root=0)
        timp_initial=time.time()
    else:
        alegere = comm.bcast(None, root=0)

    if alegere == "1":   
        # Folosim keras.datasets.cifar10.load_data pentru datele de invatare si testare:
        # imagini_invatare: 50000 de imagini de 32x32 pixeli cu valori de la 0-255
        # imagini_invatare: 10000 de etichete de la 0-9
        # imagini_testare: 10000 de imagini de 32x32 pixeli cu valori de la 0-255
        # etichete_testare: 10000 de etichete de la 0-9
        (imagini_invatare, etichete_invatare), (imagini_testare, etichete_testare) = keras.datasets.cifar10.load_data()

        # Impartim datele de invatare
        imagini_invatare_locale = imparte_date(imagini_invatare, numar_procese, rank)
        etichete_invatare_locale = imparte_date(etichete_invatare, numar_procese, rank)

        # Procesam datele de invatare
        imagini_invatare_locale = procesare_date(imagini_invatare_locale)

        # Modelul 1
        model_v1 = fun_model_v1()

        # Invatam modelul 1
        invatare_model_v1(model_v1, imagini_invatare_locale, etichete_invatare_locale)

        # Reducem gradientii pentru modelul 1
        reducere_gradienti(model_v1)

        # Procesam datele de testare
        imagini_testare_local = procesare_date(imagini_testare)

        # Testam modelul 1
        test_precizie_mv1 = model_v1.evaluate(imagini_testare_local, etichete_testare)
        rezultate_precizii.append((rank, test_precizie_mv1[1]))
    if alegere == "2":   
        (imagini_invatare, etichete_invatare), (imagini_testare, etichete_testare) = keras.datasets.cifar10.load_data()

        # Impartim datele de invatare
        imagini_invatare_locale = imparte_date(imagini_invatare, numar_procese, rank)
        etichete_invatare_locale = imparte_date(etichete_invatare, numar_procese, rank)

        # Procesam datele de invatare
        imagini_invatare_locale = procesare_date(imagini_invatare_locale)

        # Modelul 2
        model_v2 = fun_model_v2()

        # Invatam modelul 2
        invatare_model_v2(model_v2, imagini_invatare_locale, etichete_invatare_locale)

        # Reducem gradientii pentru modelul 2
        reducere_gradienti(model_v2)

        # Procesam datele de testare
        imagini_testare_local = procesare_date(imagini_testare)

        # Testam modelul 2
        test_precizie_mv2 = model_v2.evaluate(imagini_testare_local, etichete_testare)
        rezultate_precizii.append((rank, test_precizie_mv2[1]))
    if alegere == "3":   
        (imagini_invatare, etichete_invatare), (imagini_testare, etichete_testare) = keras.datasets.cifar10.load_data()

        # Impartim datele de invatare
        imagini_invatare_locale = imparte_date(imagini_invatare, numar_procese, rank)
        etichete_invatare_locale = imparte_date(etichete_invatare, numar_procese, rank)

        # Procesam datele de invatare
        imagini_invatare_locale = procesare_date(imagini_invatare_locale)

        # Modelul 3
        model_v3 = fun_model_v3()

        # Invatam modelul 3
        invatare_model_v3(model_v3, imagini_invatare_locale, etichete_invatare_locale)

        # Reducem gradientii pentru modelul 3
        reducere_gradienti(model_v3)

        # Procesam datele de testare
        imagini_testare_local = procesare_date(imagini_testare)

        # Testam modelul 3
        test_precizie_mv3 = model_v3.evaluate(imagini_testare_local, etichete_testare)
        rezultate_precizii.append((rank, test_precizie_mv3[1]))
    if alegere == "0":
        print("Procesul se inchide.")

    # Folosim Gather pentru a aduna toate rezultatele
    rezultate_precizii_adunat = comm.gather(rezultate_precizii, root=0)

    if rank == 0:
        # Afisam rezultatele
        if alegere != "0":
            print("Model", alegere)
            for rezultate in rezultate_precizii_adunat:
                for rank, precizie in rezultate:
                    print("Procesul %d - Precizie: %.4f" % (rank, precizie))
            timp_final=time.time()
            timp_total=timp_final-timp_initial
            print("Timp de procesare: %f" % (timp_total))
        elif alegere == "0":
            print("Program inchis.")