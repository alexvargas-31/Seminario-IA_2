import random as rd
import pandas as pd
import matplotlib.pyplot as plt

#*************************************************
#Punto 1
#Variables ENTRENAMIENTO1
dfP1 = pd.read_csv("spheres1d10.csv")
dfP1F = dfP1.iloc[0:160]
x1P1 = dfP1F["x1"]
x2P1 = dfP1F["x2"]
x3P1 = dfP1F["x3"]
yP1 = dfP1F["Yd"]
zP1 = []
auzP1 = []
w1P1 = []
w2P1 = []
w3P1 = []
umbralP1 = []
epocasP1 = 0
factorAprendizajeP1 = 0.3

for i in range(160):
    zP1.append(i)
    w1P1.append(rd.random())
    w2P1.append(rd.random())
    w3P1.append(rd.random())
    umbralP1.append(rd.random())
    auzP1.append(i)

#*************************************************

#*************************************************
#Funcion Entrenamiento1
def entrenamiento1(dfP1, x1P1, x2P1, x3P1, yP1, zP1, w1P1, w2P1, w3P1, umbralP1, epocasP1,
                   factorAprendizajeP1):
    cont = 0
    X = []
    for i in range(160):
        X.append(i)
    
    errores = True
    while errores:
        errores = False
        for i in range(160):
            zP1[i] = (((x1P1[i] * w1P1[i]) + (x2P1[i] * w2P1[i])
                             + (x3P1[i] * w3P1[i])) - umbralP1[i])
            # print("z ",zP1)
            auzP1.append(zP1[i])

            if zP1[i] >= 0:
                zP1[i] = 1
            else:
                zP1[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if zP1[i] != yP1[i]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yP1[i] - zP1[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralP1[i] = umbralP1[i] + (- factorAprendizajeP1 * error)
                w1P1[i] = w1P1[i] + (factorAprendizajeP1 * error * x1P1[i])
                w2P1[i] = w2P1[i] + (factorAprendizajeP1 * error * x2P1[i])
                w3P1[i] = w3P1[i] + (factorAprendizajeP1 * error * x3P1[i])

                epocasP1 += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / 160) * 100
    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Entrenamiento 1")
    print("=========================================")
    # print("w1 ",w1[0:10])
    print("", auzP1[-160:])

    # ax[0].scatter(X, auzP1[-160:], color = "b")
    # #titulo a la grafica
    # ax[0].set_title("0 - 160")
#*************************************************

#*************************************************
#Variables ENTRENAMIENTO2
dfP2 = pd.read_csv("spheres1d10.csv")
x1P2 = dfP2["x1"].iloc[160:320]
x2P2 = dfP2["x2"].iloc[160:320]
x3P2 = dfP2["x3"].iloc[160:320]
yP2 = dfP2["Yd"].iloc[160:320]
zP2 = []
auzP2 = []
w1P2 = []
w2P2 = []
w3P2 = []
umbralP2 = []
epocasP2 = 0
factorAprendizajeP2 = 0.3

for i in range(160):
    zP2.append(i)
    w1P2.append(rd.random())
    w2P2.append(rd.random())
    w3P2.append(rd.random())
    umbralP2.append(rd.random())
    auzP2.append(i)
#*************************************************

#*************************************************
#Funcion Entrenamiento2
def entrenamiento2(dfP2, x1P2, x2P2, x3P2, yP2, zP2, w1P2, w2P2, w3P2, umbralP2, epocasP2,
                   factorAprendizajeP2):
    cont = 0
    X = []
    for i in range(160):
        X.append(i)
    
    errores = True
    while errores:
        errores = False
        for i in range(160):
            zP2[i] = (((x1P2[i + 160] * w1P2[i]) + (x2P2[i + 160] * w2P2[i]) + (x3P2[i +
                                                                                     160] * w3P2[i])) - umbralP2[i])
            # print("z ",z)
            auzP2.append(zP2[i])

            if zP2[i] >= 0:
                zP2[i] = 1
            else:
                zP2[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if zP2[i] != yP2[i + 160]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yP2[i + 160] - zP2[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralP2[i] = umbralP2[i] + (- factorAprendizajeP2 * error)
                w1P2[i] = w1P2[i] + (factorAprendizajeP2 * error * x1P2[i + 160])
                w2P2[i] = w2P2[i] + (factorAprendizajeP2 * error * x2P2[i + 160])
                w3P2[i] = w3P2[i] + (factorAprendizajeP2 * error * x3P2[i + 160])

                epocasP2 += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / 160) * 100
    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Entrenamiento 2")
    print("=========================================")
    # print("w1 ",w1[0:10])
    print("", auzP2[-160:])

    # ax[1].scatter(X, auzP2[-160:], color = "b")
    # ax[1].set_title("160 - 320")
#*************************************************

#*************************************************
#Variables ENTRENAMIENTO3
dfP3 = pd.read_csv("spheres1d10.csv")
x1P3 = dfP3["x1"].iloc[320:480]
x2P3 = dfP3["x2"].iloc[320:480]
x3P3 = dfP3["x3"].iloc[320:480]
yP3 = dfP3["Yd"].iloc[320:480]
zP3 = []
auzP3 = []
w1P3 = []
w2P3 = []
w3P3 = []
umbralP3 = []
epocasP3 = 0
factorAprendizajeP3 = 0.3

for i in range(160):
    zP3.append(i)
    w1P3.append(rd.random())
    w2P3.append(rd.random())
    w3P3.append(rd.random())
    umbralP3.append(rd.random())
    auzP3.append(i)
#*************************************************

#*************************************************
#Funcion Entrenamiento3
def entrenamiento3(dfP3, x1P3, x2P3, x3P3, yP3, zP3, w1P3, w2P3, w3P3, umbralP3, epocasP3,
                   factorAprendizajeP3):
    cont = 0
    X = []
    for i in range(160):
        X.append(i)
    
    errores = True
    while errores:
        errores = False
        for i in range(160):
            zP3[i] = (((x1P3[i + 320] * w1P3[i]) + (x2P3[i + 320] * w2P3[i]) + (x3P3[i +
                                                                                     320] * w3P3[i])) - umbralP3[i])
            # print("z ",z)
            auzP3.append(zP3[i])

            if zP3[i] >= 0:
                zP3[i] = 1
            else:
                zP3[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if zP3[i] != yP3[i + 320]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yP3[i + 320] - zP3[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralP3[i] = umbralP3[i] + (- factorAprendizajeP3 * error)
                w1P3[i] = w1P3[i] + (factorAprendizajeP3 * error * x1P3[i + 320])
                w2P3[i] = w2P3[i] + (factorAprendizajeP3 * error * x2P3[i + 320])
                w3P3[i] = w3P3[i] + (factorAprendizajeP3 * error * x3P3[i + 320])

                epocasP3 += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / 160) * 100
    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Entrenamiento 3")
    print("=========================================")
    # print("w1 ",w1[0:10])
    print(" ", auzP3[-160:])

    # ax[2].scatter(X, auzP3[-160:], color = "b")
    # ax[2].set_title("320 - 480")
#*************************************************

#*************************************************
#Variables ENTRENAMIENTO4
dfP4 = pd.read_csv("spheres1d10.csv")
x1P4 = dfP4["x1"].iloc[480:640]
x2P4 = dfP4["x2"].iloc[480:640]
x3P4 = dfP4["x3"].iloc[480:640]
yP4 = dfP4["Yd"].iloc[480:640]
zP4 = []
auzP4 = []
w1P4 = []
w2P4 = []
w3P4 = []
umbralP4 = []
epocasP4 = 0
factorAprendizajeP4 = 0.3

for i in range(160):
    zP4.append(i)
    w1P4.append(rd.random())
    w2P4.append(rd.random())
    w3P4.append(rd.random())
    umbralP4.append(rd.random())
    auzP4.append(i)
#*************************************************

#*************************************************
#Funcion Entrenamiento4
def entrenamiento4(dfP4, x1P4, x2P4, x3P4, yP4, zP4, w1P4, w2P4, w3P4, umbralP4, epocasP4,
                   factorAprendizajeP4):
    cont = 0
    X = []
    for i in range(160):
        X.append(i)
    
    errores = True
    while errores:
        errores = False
        for i in range(160):
            zP4[i] = (((x1P4[i + 480] * w1P4[i]) + (x2P4[i + 480] * w2P4[i]) + (x3P4[i +
                                                                                     480] * w3P4[i])) - umbralP4[i])
            # print("z ",z)
            auzP4.append(zP4[i])

            if zP4[i] >= 0:
                zP4[i] = 1
            else:
                zP4[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if zP4[i] != yP4[i + 480]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yP4[i + 480] - zP4[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralP4[i] = umbralP4[i] + (- factorAprendizajeP4 * error)
                w1P4[i] = w1P4[i] + (factorAprendizajeP4 * error * x1P4[i + 480])
                w2P4[i] = w2P4[i] + (factorAprendizajeP4 * error * x2P4[i + 480])
                w3P4[i] = w3P4[i] + (factorAprendizajeP4 * error * x3P4[i + 480])

                epocasP4 += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / 160) * 100
    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Entrenamiento 4",)
    print("=========================================")
    # print("w1 ",w1[0:10])
    print(" ", auzP4[-160:])

    # ax[3].scatter(X, auzP4[-160:], color = "b")
    # ax[3].set_title("480 - 640")
#*************************************************

#*************************************************
#Variables ENTRENAMIENTO5
dfP5 = pd.read_csv("spheres1d10.csv")
x1P5 = dfP5["x1"].iloc[600:800]
x2P5 = dfP5["x2"].iloc[600:800]
x3P5 = dfP5["x3"].iloc[600:800]
yP5 = dfP5["Yd"].iloc[600:800]
zP5 = []
auzPi5 = []
w1P5 = []
w2P5 = []
w3P5 = []
umbralP5 = []
epocasP5 = 0
factorAprendizajeP5 = 0.3

for i in range(200):
    zP5.append(i)
    w1P5.append(rd.random())
    w2P5.append(rd.random())
    w3P5.append(rd.random())
    umbralP5.append(rd.random())
    auzPi5.append(i)
#*************************************************

#*************************************************
#Funcion Entrenamiento5
def entrenamiento5(dfP5, x1P5, x2P5, x3P5, yP5, zP5, w1P5, w2P5, w3P5, umbralP5, epocasP5,
                   factorAprendizajeP5, auzPi5):
    cont = 0
    X = []
    for i in range(200):
        X.append(i)
    
    errores = True
    while errores:
        errores = False
        for i in range(200):
            zP5[i] = (((x1P5[i + 600] * w1P5[i]) + (x2P5[i + 600] * w2P5[i]) + (x3P5[i +
                                                                                     600] * w3P5[i])) - umbralP5[i])
            # print("z ",z)
            auzPi5.append(zP5[i])

            if zP5[i] >= 0:
                zP5[i] = 1
            else:
                zP5[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if zP5[i] != yP5[i + 600]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yP5[i + 600] - zP5[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralP5[i] = umbralP5[i] + (- factorAprendizajeP5 * error)
                w1P5[i] = w1P5[i] + (factorAprendizajeP5 * error * x1P5[i + 600])
                w2P5[i] = w2P5[i] + (factorAprendizajeP5 * error * x2P5[i + 600])
                w3P5[i] = w3P5[i] + (factorAprendizajeP5 * error * x3P5[i + 600])

                epocasP5 += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / 200) * 100
    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Entrenamiento 5",)
    print("=========================================")
    # print("w1 ",w1[0:10])
    print(" ", auzPi5[-200:])

    # ax[4].scatter(X, auzPi5[-200:], color = "b")
    # ax[4].set_title("600 - 800")
#*************************************************



#*************************************************
#Generalizacion
dfG = pd.read_csv("spheres1d10.csv")
x1G = dfG["x1"].iloc[800:1000]
x2G = dfG["x2"].iloc[800:1000]
x3G = dfG["x3"].iloc[800:1000]
yG = dfG["Yd"].iloc[800:1000]
zG = []
auzG = []
factorAprendizajeG = 0.3

for i in range(200):
    zG.append(i)
    auzG.append(i)


def prueba(x1G, x2G, x3G, yG, w1G, w2G, w3G, umbralG, zG):
    XG = []
    for i in range(200):
        XG.append(i)
    
    contG = 0
    epocasG = 0
    print("***************************************")
    errores = True
    while errores:
        errores = False
        for i in range(200):
            zG[i] = (((x1G[i + 800] * w1G[i]) + (x2G[i + 800] * w2G[i]) + (x3G[i + 800] *
                                                                          w3G[i])) - umbralG[i])
            auzG.append(zG[i])

            if zG[i] >= 0:
                zG[i] = 1
            else:
                zG[i] = -1

            # print("despues z {} VS {}".format(zP[i], yP[i]))
            if zG[i] != yG[i + 800]:
                contG += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yG[i + 800] - zG[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralG[i] = umbralG[i] + (- factorAprendizajeG * error)
                w1G[i] = w1G[i] + (factorAprendizajeG * error * x1G[i + 800])
                w2G[i] = w2G[i] + (factorAprendizajeG * error * x2G[i + 800])
                w3G[i] = w3G[i] + (factorAprendizajeG * error * x3G[i + 800])

                epocasG += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1P[i])
                # print("w2 ",w2P[i])
                # print("umbral ",umbralP[i])
                # print("epocas ",epocasP)
    errorPorcentual = (1 - contG / 200) * 100

    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Generalizacion")
    print("=========================================")
    print("", auzG[-200:])

    # plt.scatter(XG, auzG[-200:])
    # plt.title("Verificacion")
#*************************************************

#*************************************************
#Punto 2
#Variables Particion 1
dfPD1 = pd.read_csv("spheres2d10.csv")
dfPDF1 = dfPD1.iloc[0:500]
x1PD1 = dfPDF1["x1"]
x2PD1 = dfPDF1["x2"]
x3PD1 = dfPDF1["x3"]
yPD1 = dfPDF1["Yd"]
zPD1 = []
auzPD1 = []
w1PD1 = []
w2PD1 = []
w3PD1 = []
umbralPD1 = []
epocasPD1 = 0
factorAprendizajePD1 = 0.3

for i in range(500):
    zPD1.append(i)
    w1PD1.append(rd.random())
    w2PD1.append(rd.random())
    w3PD1.append(rd.random())
    umbralPD1.append(rd.random())
    auzPD1.append(i)

#*************************************************

#*************************************************
#Funcion Entrenamiento1
def particion1(x1PD1, x2PD1, x3PD1, yPD1, zPD1, w1PD1, w2PD1, w3PD1, umbralPD1,
               epocasPD1, factorAprendizajePD1):
    cont = 0
    X = []
    for i in range(500):
        X.append(i)
    
    errores = True
    while errores:
        errores = False
        for i in range(500):
            zPD1[i] = (((x1PD1[i] * w1PD1[i]) + (x2PD1[i] * w2PD1[i])
                             + (x3PD1[i] * w3PD1[i])) - umbralPD1[i])
            # print("z ",zPD1)
            auzPD1.append(zPD1[i])

            if zPD1[i] >= 0:
                zPD1[i] = 1
            else:
                zPD1[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if zPD1[i] != yPD1[i]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yPD1[i] - zPD1[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralPD1[i] = umbralPD1[i] + (- factorAprendizajePD1 * error)
                w1PD1[i] = w1PD1[i] + (factorAprendizajePD1 * error * x1PD1[i])
                w2PD1[i] = w2PD1[i] + (factorAprendizajePD1 * error * x2PD1[i])
                w3PD1[i] = w3PD1[i] + (factorAprendizajePD1 * error * x3PD1[i])

                epocasPD1 += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / 500) * 100
    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Particion 1")
    print("=========================================")
    # print("w1 ",w1[0:10])
    print("", auzPD1[-500:])
#************************************************* 

#*************************************************
#Punto 2
#Variables Particion 2
dfPDF2 = dfPD1.iloc[500:1000]
x1PD2 = dfPDF2["x1"]
x2PD2 = dfPDF2["x2"]
x3PD2 = dfPDF2["x3"]
yPD2 = dfPDF2["Yd"]
zPD2 = []
auzPD2 = []
w1PD2 = []
w2PD2 = []
w3PD2 = []
umbralPD2 = []
epocasPD2 = 0
factorAprendizajePD2 = 0.3

for i in range(500):
    zPD2.append(i)
    w1PD2.append(rd.random())
    w2PD2.append(rd.random())
    w3PD2.append(rd.random())
    umbralPD2.append(rd.random())
    auzPD2.append(i)
#*************************************************

#*************************************************
#Funcion Entrenamiento1
def particion2(x1PD2, x2PD2, x3PD2, yPD2, zPD2, w1PD2, w2PD2, w3PD2, umbralPD2,
               epocasPD2, factorAprendizajePD2):
    cont = 0
    X = []
    for i in range(500):
        X.append(i)
    
    errores = True
    while errores:
        errores = False
        for i in range(500):
            zPD2[i] = (((x1PD2[i + 500] * w1PD2[i]) + (x2PD2[i + 500] * w2PD2[i])
                             + (x3PD2[i + 500] * w3PD2[i])) - umbralPD2[i])
            # print("z ",zPD2)
            auzPD2.append(zPD2[i])

            if zPD2[i] >= 0:
                zPD2[i] = 1
            else:
                zPD2[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if zPD2[i] != yPD2[i + 500]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yPD2[i + 500] - zPD2[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralPD2[i] = umbralPD2[i] + (- factorAprendizajePD2 * error)
                w1PD2[i] = w1PD2[i] + (factorAprendizajePD2 * error * x1PD2[i + 500])
                w2PD2[i] = w2PD2[i] + (factorAprendizajePD2 * error * x2PD2[i + 500])
                w3PD2[i] = w3PD2[i] + (factorAprendizajePD2 * error * x3PD2[i + 500])

                epocasPD2 += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / 500) * 100
    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Particion 2")
    print("=========================================")
    # print("w1 ",w1[0:10])
    print("", auzPD2[-500:])
#************************************************* 

#*************************************************
#Punto 2
#Variables Particion 3
dfPDF3 = dfPD1.iloc[1000:1500]
x1PD3 = dfPDF3["x1"]
x2PD3 = dfPDF3["x2"]
x3PD3 = dfPDF3["x3"]
yPD3 = dfPDF3["Yd"]
zPD3 = []
auzPD3 = []
w1PD3 = []
w2PD3 = []
w3PD3 = []
umbralPD3 = []
epocasPD3 = 0
factorAprendizajePD3 = 0.3

for i in range(500):
    zPD3.append(i)
    w1PD3.append(rd.random())
    w2PD3.append(rd.random())
    w3PD3.append(rd.random())
    umbralPD3.append(rd.random())
    auzPD3.append(i)
#*************************************************

#*************************************************
#Funcion Entrenamiento3
def particion3(x1PD3, x2PD3, x3PD3, yPD3, zPD3, w1PD3, w2PD3, w3PD3, umbralPD3,
               epocasPD3, factorAprendizajePD3):
    cont = 0
    X = []
    for i in range(500):
        X.append(i)
    
    errores = True
    while errores:
        errores = False
        for i in range(500):
            zPD3[i] = (((x1PD3[i + 1000] * w1PD3[i]) + (x2PD3[i + 1000] * w2PD3[i])
                             + (x3PD3[i + 1000] * w3PD3[i])) - umbralPD3[i])
            # print("z ",zPD3)
            auzPD3.append(zPD3[i])

            if zPD3[i] >= 0:
                zPD3[i] = 1
            else:
                zPD3[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if zPD3[i] != yPD3[i + 1000]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yPD3[i + 1000] - zPD3[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralPD3[i] = umbralPD3[i] + (- factorAprendizajePD3 * error)
                w1PD3[i] = w1PD3[i] + (factorAprendizajePD3 * error * x1PD3[i + 1000])
                w2PD3[i] = w2PD3[i] + (factorAprendizajePD3 * error * x2PD3[i + 1000])
                w3PD3[i] = w3PD3[i] + (factorAprendizajePD3 * error * x3PD3[i + 1000])

                epocasPD3 += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / 500) * 100
    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Particion 3")
    print("=========================================")
    # print("w1 ",w1[0:10])
    print("", auzPD3[-500:])
#*************************************************]

#*************************************************
#Punto 2
#Variables Particion 4
dfPDF4 = dfPD1.iloc[1500:2000]
x1PD4 = dfPDF4["x1"]
x2PD4 = dfPDF4["x2"]
x3PD4 = dfPDF4["x3"]
yPD4 = dfPDF4["Yd"]
zPD4 = []
auzPD4 = []
w1PD4 = []
w2PD4 = []
w3PD4 = []
umbralPD4 = []
epocasPD4 = 0
factorAprendizajePD4 = 0.3

for i in range(500):
    zPD4.append(i)
    w1PD4.append(rd.random())
    w2PD4.append(rd.random())
    w3PD4.append(rd.random())
    umbralPD4.append(rd.random())
    auzPD4.append(i)
#*************************************************

#*************************************************
#Funcion Entrenamiento4
def particion4(x1PD4, x2PD4, x3PD4, yPD4, zPD4, w1PD4, w2PD4, w3PD4, umbralPD4,
               epocasPD4, factorAprendizajePD4):
    cont = 0
    X = []
    for i in range(500):
        X.append(i)
    
    errores = True
    while errores:
        errores = False
        for i in range(500):
            zPD4[i] = (((x1PD4[i + 1500] * w1PD4[i]) + (x2PD4[i + 1500] * w2PD4[i])
                             + (x3PD4[i + 1500] * w3PD4[i])) - umbralPD4[i])
            # print("z ",zPD4)
            auzPD4.append(zPD4[i])

            if zPD4[i] >= 0:
                zPD4[i] = 1
            else:
                zPD4[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if zPD4[i] != yPD4[i + 1500]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yPD4[i + 1500] - zPD4[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralPD4[i] = umbralPD4[i] + (- factorAprendizajePD4 * error)
                w1PD4[i] = w1PD4[i] + (factorAprendizajePD4 * error * x1PD4[i + 1500])
                w2PD4[i] = w2PD4[i] + (factorAprendizajePD4 * error * x2PD4[i + 1500])
                w3PD4[i] = w3PD4[i] + (factorAprendizajePD4 * error * x3PD4[i + 1500])

                epocasPD4 += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / 500) * 100
    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Particion 4")
    print("=========================================")
    # print("w1 ",w1[0:10])
    print("", auzPD4[-500:])
#*************************************************]

#*************************************************
#Punto 2
#Variables Particion 5
dfPDF5 = dfPD1.iloc[2000:2500]
x1PD5 = dfPDF5["x1"]
x2PD5 = dfPDF5["x2"]
x3PD5 = dfPDF5["x3"]
yPD5 = dfPDF5["Yd"]
zPD5 = []
auzPD5 = []
w1PD5 = []
w2PD5 = []
w3PD5 = []
umbralPD5 = []
epocasPD5 = 0
factorAprendizajePD5 = 0.3

for i in range(500):
    zPD5.append(i)
    w1PD5.append(rd.random())
    w2PD5.append(rd.random())
    w3PD5.append(rd.random())
    umbralPD5.append(rd.random())
    auzPD5.append(i)
#*************************************************

#*************************************************
#Funcion Entrenamiento5
def particion5(x1PD5, x2PD5, x3PD5, yPD5, zPD5, w1PD5, w2PD5, w3PD5, umbralPD5,
               epocasPD5, factorAprendizajePD5):
    cont = 0
    X = []
    for i in range(500):
        X.append(i)
    
    errores = True
    while errores:
        errores = False
        for i in range(500):
            zPD5[i] = (((x1PD5[i + 2000] * w1PD5[i]) + (x2PD5[i + 2000] * w2PD5[i])
                             + (x3PD5[i + 2000] * w3PD5[i])) - umbralPD5[i])
            # print("z ",zPD5)
            auzPD5.append(zPD5[i])

            if zPD5[i] >= 0:
                zPD5[i] = 1
            else:
                zPD5[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if zPD5[i] != yPD5[i + 2000]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yPD5[i + 2000] - zPD5[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralPD5[i] = umbralPD5[i] + (- factorAprendizajePD5 * error)
                w1PD5[i] = w1PD5[i] + (factorAprendizajePD5 * error * x1PD5[i + 2000])
                w2PD5[i] = w2PD5[i] + (factorAprendizajePD5 * error * x2PD5[i + 2000])
                w3PD5[i] = w3PD5[i] + (factorAprendizajePD5 * error * x3PD5[i + 2000])

                epocasPD5 += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / 500) * 100
    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Particion 5")
    print("=========================================")
    # print("w1 ",w1[0:10])
    print("", auzPD5[-500:])
#*************************************************]

#*************************************************
#Punto 2
#Variables Particion 6
dfPDF6 = dfPD1.iloc[2500:3000]
x1PD6 = dfPDF6["x1"]
x2PD6 = dfPDF6["x2"]
x3PD6 = dfPDF6["x3"]
yPD6 = dfPDF6["Yd"]
zPD6 = []
auzPD6 = []
w1PD6 = []
w2PD6 = []
w3PD6 = []
umbralPD6 = []
epocasPD6 = 0
factorAprendizajePD6 = 0.3

for i in range(500):
    zPD6.append(i)
    w1PD6.append(rd.random())
    w2PD6.append(rd.random())
    w3PD6.append(rd.random())
    umbralPD6.append(rd.random())
    auzPD6.append(i)
#*************************************************

#*************************************************
#Funcion Entrenamiento6
def particion6(x1PD6, x2PD6, x3PD6, yPD6, zPD6, w1PD6, w2PD6, w3PD6, umbralPD6,
               epocasPD6, factorAprendizajePD6):
    cont = 0
    X = []
    for i in range(500):
        X.append(i)
    
    errores = True
    while errores:
        errores = False
        for i in range(500):
            zPD6[i] = (((x1PD6[i + 2500] * w1PD6[i]) + (x2PD6[i + 2500] * w2PD6[i])
                             + (x3PD6[i + 2500] * w3PD6[i])) - umbralPD6[i])
            # print("z ",zPD6)
            auzPD6.append(zPD6[i])

            if zPD6[i] >= 0:
                zPD6[i] = 1
            else:
                zPD6[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if zPD6[i] != yPD6[i + 2500]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yPD6[i + 2500] - zPD6[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralPD6[i] = umbralPD6[i] + (- factorAprendizajePD6 * error)
                w1PD6[i] = w1PD6[i] + (factorAprendizajePD6 * error * x1PD6[i + 2500])
                w2PD6[i] = w2PD6[i] + (factorAprendizajePD6 * error * x2PD6[i + 2500])
                w3PD6[i] = w3PD6[i] + (factorAprendizajePD6 * error * x3PD6[i + 2500])

                epocasPD6 += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / 500) * 100
    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Particion 6")
    print("=========================================")
    # print("w1 ",w1[0:10])
    print("", auzPD6[-500:])
#*************************************************]

#*************************************************
#Punto 2
#Variables Particion 7
dfPDF7 = dfPD1.iloc[3000:3500]
x1PD7 = dfPDF7["x1"]
x2PD7 = dfPDF7["x2"]
x3PD7 = dfPDF7["x3"]
yPD7 = dfPDF7["Yd"]
zPD7 = []
auzPD7 = []
w1PD7 = []
w2PD7 = []
w3PD7 = []
umbralPD7 = []
epocasPD7 = 0
factorAprendizajePD7 = 0.3

for i in range(500):
    zPD7.append(i)
    w1PD7.append(rd.random())
    w2PD7.append(rd.random())
    w3PD7.append(rd.random())
    umbralPD7.append(rd.random())
    auzPD7.append(i)
#*************************************************

#*************************************************
#Funcion Entrenamiento7
def particion7(x1PD7, x2PD7, x3PD7, yPD7, zPD7, w1PD7, w2PD7, w3PD7, umbralPD7,
               epocasPD7, factorAprendizajePD7):
    cont = 0
    X = []
    for i in range(500):
        X.append(i)
    
    errores = True
    while errores:
        errores = False
        for i in range(500):
            zPD7[i] = (((x1PD7[i + 3000] * w1PD7[i]) + (x2PD7[i + 3000] * w2PD7[i])
                             + (x3PD7[i + 3000] * w3PD7[i])) - umbralPD7[i])
            # print("z ",zPD7)
            auzPD7.append(zPD7[i])

            if zPD7[i] >= 0:
                zPD7[i] = 1
            else:
                zPD7[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if zPD7[i] != yPD7[i + 3000]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yPD7[i + 3000] - zPD7[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralPD7[i] = umbralPD7[i] + (- factorAprendizajePD7 * error)
                w1PD7[i] = w1PD7[i] + (factorAprendizajePD7 * error * x1PD7[i + 3000])
                w2PD7[i] = w2PD7[i] + (factorAprendizajePD7 * error * x2PD7[i + 3000])
                w3PD7[i] = w3PD7[i] + (factorAprendizajePD7 * error * x3PD7[i + 3000])

                epocasPD7 += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / 500) * 100
    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Particion 7")
    print("=========================================")
    # print("w1 ",w1[0:10])
    print("", auzPD7[-500:])
#*************************************************]

#*************************************************
#Punto 2
#Variables Particion 8
dfPDF8 = dfPD1.iloc[3500:4000]
x1PD8 = dfPDF8["x1"]
x2PD8 = dfPDF8["x2"]
x3PD8 = dfPDF8["x3"]
yPD8 = dfPDF8["Yd"]
zPD8 = []
auzPD8 = []
w1PD8 = []
w2PD8 = []
w3PD8 = []
umbralPD8 = []
epocasPD8 = 0
factorAprendizajePD8 = 0.3

for i in range(500):
    zPD8.append(i)
    w1PD8.append(rd.random())
    w2PD8.append(rd.random())
    w3PD8.append(rd.random())
    umbralPD8.append(rd.random())
    auzPD8.append(i)
#*************************************************

#*************************************************
#Funcion Entrenamiento8
def particion8(x1PD8, x2PD8, x3PD8, yPD8, zPD8, w1PD8, w2PD8, w3PD8, umbralPD8,
               epocasPD8, factorAprendizajePD8):
    cont = 0
    X = []
    for i in range(500):
        X.append(i)
    
    errores = True
    while errores:
        errores = False
        for i in range(500):
            zPD8[i] = (((x1PD8[i + 3500] * w1PD8[i]) + (x2PD8[i + 3500] * w2PD8[i])
                             + (x3PD8[i + 3500] * w3PD8[i])) - umbralPD8[i])
            # print("z ",zPD8)
            auzPD8.append(zPD8[i])

            if zPD8[i] >= 0:
                zPD8[i] = 1
            else:
                zPD8[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if zPD8[i] != yPD8[i + 3500]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yPD8[i + 3500] - zPD8[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralPD8[i] = umbralPD8[i] + (- factorAprendizajePD8 * error)
                w1PD8[i] = w1PD8[i] + (factorAprendizajePD8 * error * x1PD8[i + 3500])
                w2PD8[i] = w2PD8[i] + (factorAprendizajePD8 * error * x2PD8[i + 3500])
                w3PD8[i] = w3PD8[i] + (factorAprendizajePD8 * error * x3PD8[i + 3500])

                epocasPD8 += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / 500) * 100
    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Particion 8")
    print("=========================================")
    # print("w1 ",w1[0:10])
    print("", auzPD8[-500:])
#*************************************************]

#*************************************************
#Punto 2
#Variables Particion 9
dfPDF9 = dfPD1.iloc[4000:4500]
x1PD9 = dfPDF9["x1"]
x2PD9 = dfPDF9["x2"]
x3PD9 = dfPDF9["x3"]
yPD9 = dfPDF9["Yd"]
zPD9 = []
auzPD9 = []
w1PD9 = []
w2PD9 = []
w3PD9 = []
umbralPD9 = []
epocasPD9 = 0
factorAprendizajePD9 = 0.3

for i in range(500):
    zPD9.append(i)
    w1PD9.append(rd.random())
    w2PD9.append(rd.random())
    w3PD9.append(rd.random())
    umbralPD9.append(rd.random())
    auzPD9.append(i)
#*************************************************

#*************************************************
#Funcion Entrenamiento9
def particion9(x1PD9, x2PD9, x3PD9, yPD9, zPD9, w1PD9, w2PD9, w3PD9, umbralPD9,
               epocasPD9, factorAprendizajePD9):
    cont = 0
    X = []
    for i in range(500):
        X.append(i)
    
    errores = True
    while errores:
        errores = False
        for i in range(500):
            zPD9[i] = (((x1PD9[i + 4000] * w1PD9[i]) + (x2PD9[i + 4000] * w2PD9[i])
                             + (x3PD9[i + 4000] * w3PD9[i])) - umbralPD9[i])
            # print("z ",zPD9)
            auzPD9.append(zPD9[i])

            if zPD9[i] >= 0:
                zPD9[i] = 1
            else:
                zPD9[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if zPD9[i] != yPD9[i + 4000]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yPD9[i + 4000] - zPD9[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralPD9[i] = umbralPD9[i] + (- factorAprendizajePD9 * error)
                w1PD9[i] = w1PD9[i] + (factorAprendizajePD9 * error * x1PD9[i + 4000])
                w2PD9[i] = w2PD9[i] + (factorAprendizajePD9 * error * x2PD9[i + 4000])
                w3PD9[i] = w3PD9[i] + (factorAprendizajePD9 * error * x3PD9[i + 4000])

                epocasPD9 += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / 500) * 100
    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Particion 9")
    print("=========================================")
    # print("w1 ",w1[0:10])
    print("", auzPD9[-500:])
#*************************************************]

#*************************************************
#Punto 2
#Variables Particion 10
dfPDF10 = dfPD1.iloc[4500:5000]
x1PD10 = dfPDF10["x1"]
x2PD10 = dfPDF10["x2"]
x3PD10 = dfPDF10["x3"]
yPD10 = dfPDF10["Yd"]
zPD10 = []
auzPD10 = []
w1PD10 = []
w2PD10 = []
w3PD10 = []
umbralPD10 = []
epocasPD10 = 0
factorAprendizajePD10 = 0.3

for i in range(500):
    zPD10.append(i)
    w1PD10.append(rd.random())
    w2PD10.append(rd.random())
    w3PD10.append(rd.random())
    umbralPD10.append(rd.random())
    auzPD10.append(i)
#*************************************************

#*************************************************
#Funcion Entrenamiento10
def particion10(x1PD10, x2PD10, x3PD10, yPD10, zPD10, w1PD10, w2PD10, w3PD10, umbralPD10,
               epocasPD10, factorAprendizajePD10):
    cont = 0
    X = []
    for i in range(500):
        X.append(i)
    
    errores = True
    while errores:
        errores = False
        for i in range(500):
            zPD10[i] = (((x1PD10[i + 4500] * w1PD10[i]) + (x2PD10[i + 4500] * w2PD10[i])
                             + (x3PD10[i + 4500] * w3PD10[i])) - umbralPD10[i])
            # print("z ",zPD10)
            auzPD10.append(zPD10[i])

            if zPD10[i] >= 0:
                zPD10[i] = 1
            else:
                zPD10[i] = -1

            # print("despues z {} VS {}".format(z[i], y[i]))
            if zPD10[i] != yPD10[i + 4500]:
                cont += 1
                # print("****************Entro a entrenar****************")
                errores = True
                error = yPD10[i + 4500] - zPD10[i]
                if error == -2:
                    error += 1
                else:
                    error -= 1

                umbralPD10[i] = umbralPD10[i] + (- factorAprendizajePD10 * error)
                w1PD10[i] = w1PD10[i] + (factorAprendizajePD10 * error * x1PD10[i + 4500])
                w2PD10[i] = w2PD10[i] + (factorAprendizajePD10 * error * x2PD10[i + 4500])
                w3PD10[i] = w3PD10[i] + (factorAprendizajePD10 * error * x3PD10[i + 4500])

                epocasPD10 += 1

                # print("Pesos Cambiados")
                # print("w1 ",w1[i])
                # print("w2 ",w2[i])
                # print("umbral ",umbral[i])
                # print("epocas ",epocas)
    errorPorcentual = (1 - cont / 500) * 100
    print("=========================================")
    print("errorPorcentual {} %".format(errorPorcentual))
    print("Particion 10")
    print("=========================================")
    # print("w1 ",w1[0:10])
    print("", auzPD10[-500:])
#*************************************************]







#*************************************************
#Entrenamientos

#*************************************************
#Variables para hacer la prueba
w1G = w1P5[0:200]
w2G = w2P5[0:200]
w3G = w3P5[0:200]
umbralG = umbralP5[0:200]
#*************************************************

#*************************************************
# #PRUEBA
#*************************************************

#*************************************************
#Particiones
opc = 3

while opc != 0:
    print("1.- Crear 5 particiones de entrenamiento usando 80 de los datos y 20 para la generalizacion")
    print("2.- Crear 10 particiones de entrenamiento usando 80 de los datos y 20 para la generalizacion")
    print("0.- Salir")
    opc = int(input("Ingrese una opcion: "))

    if opc == 1:
        # Variables para poder graficar 
        X1 = dfP1["x1"].iloc[0:160]
        X2 = dfP1["x1"].iloc[0:160]
        X3 = dfP1["x1"].iloc[0:160]

        #Graficar en 3D 800 datos
        axesE = plt.axes(projection = "3d")
        axesE.scatter3D(X1, X2, X3, color = "b")
        axesE.set_title("Entrenamiento Datos 80%")
        axesE.set_xlabel("X")
        axesE.set_ylabel("Y")
        axesE.set_zlabel("Z")
        plt.show()
        entrenamiento1(dfP1, x1P1, x2P1, x3P1, yP1, zP1, w1P1, w2P1, w3P1, umbralP1, epocasP1,
                           factorAprendizajeP1)
        entrenamiento2(dfP2, x1P2, x2P2, x3P2, yP2, zP2, w1P2, w2P2, w3P2, umbralP2, epocasP2,
                           factorAprendizajeP2)
        entrenamiento3(dfP3, x1P3, x2P3, x3P3, yP3, zP3, w1P3, w2P3, w3P3, umbralP3, epocasP3,
                           factorAprendizajeP3)
        entrenamiento4(dfP4, x1P4, x2P4, x3P4, yP4, zP4, w1P4, w2P4, w3P4, umbralP4, epocasP4,
                           factorAprendizajeP4)
        entrenamiento5(dfP5, x1P5, x2P5, x3P5, yP5, zP5, w1P5, w2P5, w3P5, umbralP5, epocasP5,
                           factorAprendizajeP5, auzPi5)
        prueba(x1G, x2G, x3G, yG, w1G, w2G, w3G, umbralG, zG)

        # Variables para poder graficar 
        XG1 = dfP1["x1"].iloc[800:1000]
        XG2 = dfP1["x1"].iloc[800:1000]
        XG3 = dfP1["x1"].iloc[800:1000]

        #Graficar en 3D 800 datos
        axesG = plt.axes(projection = "3d")
        axesG.scatter3D(X1, X2, X3, color = "r")
        axesG.set_title("Generalizacion Datos 20%")
        axesG.set_xlabel("X")
        axesG.set_ylabel("Y")
        axesG.set_zlabel("Z")
        plt.show()
    elif opc == 2:
        particion1(x1PD1, x2PD1, x3PD1, yPD1, zPD1, w1PD1, w2PD1, w3PD1, umbralPD1,
                       epocasPD1, factorAprendizajePD1)
        particion2(x1PD2, x2PD2, x3PD2, yPD2, zPD2, w1PD2, w2PD2, w3PD2, umbralPD2,
                       epocasPD2, factorAprendizajePD2)
        particion3(x1PD3, x2PD3, x3PD3, yPD3, zPD3, w1PD3, w2PD3, w3PD3, umbralPD3,
                       epocasPD3, factorAprendizajePD3)
        particion4(x1PD4, x2PD4, x3PD4, yPD4, zPD4, w1PD4, w2PD4, w3PD4, umbralPD4,
                       epocasPD4, factorAprendizajePD4)
        particion5(x1PD5, x2PD5, x3PD5, yPD5, zPD5, w1PD5, w2PD5, w3PD5, umbralPD5,
                       epocasPD5, factorAprendizajePD5)
        particion6(x1PD6, x2PD6, x3PD6, yPD6, zPD6, w1PD6, w2PD6, w3PD6, umbralPD6,
                       epocasPD6, factorAprendizajePD6)
        particion7(x1PD7, x2PD7, x3PD7, yPD7, zPD7, w1PD7, w2PD7, w3PD7, umbralPD7,
                       epocasPD7, factorAprendizajePD7)
        particion8(x1PD8, x2PD8, x3PD8, yPD8, zPD8, w1PD8, w2PD8, w3PD8, umbralPD8,
                       epocasPD8, factorAprendizajePD8)
        particion9(x1PD9, x2PD9, x3PD9, yPD9, zPD9, w1PD9, w2PD9, w3PD9, umbralPD9,
                       epocasPD9, factorAprendizajePD9)
        particion10(x1PD10, x2PD10, x3PD10, yPD10, zPD10, w1PD10, w2PD10, w3PD10, umbralPD10,
                       epocasPD10, factorAprendizajePD10)
            
        # # Variables para poder graficar 
        XD1 = dfPD1["x1"].iloc[0:500]
        XD2 = dfPD1["x2"].iloc[0:500]
        XD3 = dfPD1["x3"].iloc[0:500]

        #Graficar en 3D 800 datos
        axesD = plt.axes(projection = "3d")
        axesD.scatter3D(XD1, XD2, XD3, color = "b")
        axesD.set_title("Particion Datos 80%")
        axesD.set_xlabel("X")
        axesD.set_ylabel("Y")
        axesD.set_zlabel("Z")
        plt.show()
