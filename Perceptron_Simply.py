#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 21:44:28 2023

@author: alejandrovargas
"""

import random as rd
import pandas as pd
import matplotlib.pyplot as plt

#ENTRENAMIENTO
df = pd.read_csv("XOR_trn.csv")
x1 = df["Input1"]
x2 = df["Input2"]
y = df["Output"]
z = []
auz = []
w1 = []
w2 = []
umbral = []
epocas = 0
factorAprendizaje = 0.7

for i in range(2000):
    z.append(i)
    w1.append(rd.random())
    w2.append(rd.random())
    umbral.append(rd.random())
    auz.append(i)

#PRUEBA
dfP = pd.read_csv("XOR_tst.csv")
x1P = dfP["Input1"]
x2P = dfP["Input2"]
yP = dfP["Output"]
zP = []
auzP = []

for i in range(200):
    zP.append(i)
    auzP.append(i)


def entrenamiento(df, x1, x2, y, z, w1, w2, umbral, epocas, factorAprendizaje):
    cont = 0
    dfP = df.iloc[0:100]
    X = df["Input1"] 
    Y = df["Input2"] 
    # print("x1 ",X)
    # print("x2 ",Y)

    #*************************************************
    #PRUEBAS
    out = dfP["Output"]

    inP1 = dfP["Input1"]
    inP2 = dfP["Input2"]

    plt.scatter(inP1, inP2)

    #*************************************************
    # fig, ax = plt.subplots(1,2)

    # ax[0].plot(X, Y, marker = "v", linestyle = " ")

    errores = True
    while errores:
        errores = False
        for i in range(2000):
            z[i] = (((x1[i] * w1[i]) + (x2[i] * w2[i])) - umbral[i])
            # print("z ",z)
            auz.append(z[i])

            if z[i] >= 0:
                z[i] = 1
            else:
                z[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if z[i] != y[i]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = y[i] - z[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbral[i] = umbral[i] + (- factorAprendizaje * error)
                w1[i] = w1[i] + (factorAprendizaje * error * x1[i])
                w2[i] = w2[i] + (factorAprendizaje * error * x2[i])

                epocas += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / len(X)) * 100
    print("errorPorcentual {} %".format(errorPorcentual))
    # print("w1 ",w1[0:10])
    print("Entrenamiento ",z[-2000:])
    # ax[1].plot(X, auz[-2000:], marker = "o", linestyle = " ")

    #*************************************************
    #PRUEBAS
    # print("auz ",auz[-100:])
    plt.scatter(inP1, auz[-100:], color = "r")

    # plt.plot(inP1, inP2)

    #*************************************************

    plt.show()

def prueba(dfP, x1P, x2P, yP, w1P, w2P, umbralP):
    contP = 0
    epocasP = 0
    print("***************************************")
    errores = True
    while errores:
        errores = False
        for i in range(200):
            zP[i] = (((x1P[i] * w1P[i]) + (x2P[i]* w2P[i])) - umbralP[i])
            auzP.append(zP[i])

            if zP[i] >= 0:
                zP[i] = 1
            else:
                zP[i] = -1

            # print("despues z {} VS {}".format(zP[i], yP[i]))
            if zP[i] != yP[i]:
                contP += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yP[i] - zP[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralP[i] = umbralP[i] + (- factorAprendizaje * error)
                w1P[i] = w1P[i] + (factorAprendizaje * error * x1P[i])
                w2P[i] = w2P[i] + (factorAprendizaje * error * x2P[i])

                epocasP += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1P[i])
                # print("w2 ",w2P[i])
                # print("umbral ",umbralP[i])
                # print("epocas ",epocasP)
    errorPorcentual = (1 - contP / len(x1P)) * 100
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Verificacion ",zP[-200:])
            

#ENTRENAMIENTO
# print("****************Datos Iniciales****************")
# print("w1 ",w1)
# print("w2 ",w2)
# print("umbral ",umbral)
entrenamiento(df, x1, x2, y, z, w1, w2, umbral, epocas, factorAprendizaje)
# print("****************Datos Finales****************")
# print("w1 ",w1)
# print("w2 ",w2)
# print("umbral ",umbral)
# print("auz ",auz[-6:])


w1P = w1[0:200]
w2P = w1[0:200]
umbralP = umbral[0:200]

#PRUEBA
prueba(dfP, x1P, x2P, yP, w1P, w2P, umbralP)