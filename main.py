import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sigfig import round

with open("AM_data.txt") as f:
    data_AM = f.read().split('\n')

with open("Cd_data.txt") as f:
    data_Cd = f.read().split('\n')

with open("Co_data.txt") as f:
    data_Co = f.read().split('\n')

y = []
for i in range(len(data_Cd)):
    y += [int(data_Cd[i])]
x = np.linspace(0, len(y), len(y))


def gaussienne(x, a, sigma, mu):
    return a*np.exp(-((x-mu)/sigma)**2)

def gauss_fit(x_data, y_data, pos):
    popt, pcov = curve_fit(gaussienne, x_data, y_data, p0=[np.max(y_data), np.sqrt(np.std(y_data)), pos])
    print(popt)

    fit = gaussienne(x_data, *popt)
    ### Calcul du R^2 du fit ###
    residuals = fit - y_data
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data-np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)

    plt.plot(x_data, y_data, label="spectre Cd")
    plt.plot(x_data, fit, 
            label=f"$Curve$ $fit$ d'une gaussienne à {round(popt[2], sigfigs=3)} KeV\n$\sigma$={round(popt[1], sigfigs=3)}\n$R^2$ = {round(r_squared, sigfigs=3)}")
    plt.xlabel('Énergie [KeV]')
    plt.ylabel('Nombre de compte [-]')
    plt.legend()
    plt.show()
    return popt, r_squared


def lin(x, a, b):
    return a*x+b

popt, pcov = curve_fit(lin, [354.04, 776.89, 6506.70], [7.65, 14.4, 122.1], p0=[0.02, 0])
print(popt)


fit = lin(np.array([354.04, 776.89, 6506.70]), *popt)
residuals = fit - [7.65, 14.4, 122.1]
ss_res = np.sum(residuals**2)
ss_tot = np.sum(([7.65, 14.4, 122.1]-np.mean(np.array([7.65, 14.4, 122.1])))**2)
r_squaredfit = 1 - (ss_res / ss_tot)
print(r_squaredfit)

x = lin(x, *popt) * 4
plt.plot(x, y)
plt.show()


if False:
    x_1, y_1 = x[290:410], y[290:410]
    pos_1 = 6.7
    fit_1, r_squared1 = gauss_fit(x_1, y_1, pos_1)

    x_2, y_2 = x[730:840], y[730:840]
    pos_2 = 14
    fit_2, r_squared2 = gauss_fit(x_2, y_2, pos_2)

    x_3, y_3 = x[6480:6650], y[6480:6650]
    pos_3 = 124.5
    fit_3, r_squared3 = gauss_fit(x_3, y_3, pos_3)
    
    plt.plot(x, y, linewidth = 0.4, label="Pic d'intéret")
    plt.plot(x, gaussienne(x, *fit_1), linewidth = 0.9,
            label=f"$Curve$ $fit$ d'une gaussienne à {round(fit_1[2], sigfigs=3)} KeV\n$\sigma$={round(fit_1[1], sigfigs=3)}\n$R^2$ = {round(r_squared1, sigfigs=3)}")
    plt.plot(x, gaussienne(x, *fit_2), linewidth = 0.9, 
            label=f"$Curve$ $fit$ d'une gaussienne à {round(fit_2[2], sigfigs=3)} KeV\n$\sigma$={round(fit_2[1], sigfigs=3)}\n$R^2$ = {round(r_squared2, sigfigs=3)}")
    plt.plot(x, gaussienne(x, *fit_3), linewidth = 0.9, 
            label=f"$Curve$ $fit$ d'une gaussienne à {round(fit_3[2], sigfigs=3)} KeV\n$\sigma$={round(fit_3[1], sigfigs=3)}\n$R^2$ = {round(r_squared3, sigfigs=3)}")
    plt.legend()
    plt.show()
    


if False:
    x_1, y_1 = x[710:790], y[710:790]
    pos_1 = 14.4
    fit_1, r_squared1 = gauss_fit(x_1, y_1, pos_1)

    x_2, y_2 = x[890:1030], y[890:1030]
    pos_2 = 18.2
    fit_2, r_squared2 = gauss_fit(x_2, y_2, pos_2)

    x_3, y_3 = x[2940:3290], y[2940:3290]
    pos_3 = 60.6
    fit_3, r_squared3 = gauss_fit(x_3, y_3, pos_3)
    
    plt.plot(x, y, linewidth = 0.4, label="spectre de l'américium")
    plt.plot(x, gaussienne(x, *fit_1), linewidth = 0.9, 
            label=f"$Curve$ $fit$ d'une gaussienne à {round(fit_1[2], sigfigs=3)} KeV\n$\sigma$={round(fit_1[1], sigfigs=3)}\n$R^2$ = {round(r_squared1, sigfigs=3)}")
    plt.plot(x, gaussienne(x, *fit_2), linewidth = 0.9, 
            label=f"$Curve$ $fit$ d'une gaussienne à {round(fit_2[2], sigfigs=3)} KeV\n$\sigma$={round(fit_2[1], sigfigs=3)}\n$R^2$ = {round(r_squared2, sigfigs=3)}")
    plt.plot(x, gaussienne(x, *fit_3), linewidth = 0.9, 
            label=f"$Curve$ $fit$ d'une gaussienne à {round(fit_3[2], sigfigs=3)} KeV\n$\sigma$={round(fit_3[1], sigfigs=3)}\n$R^2$ = {round(r_squared3, sigfigs=3)}")
    plt.legend()
    plt.show()
    
res2 = [0.891, 1.107, 1.53]
pics2 = [14.6, 18.4, 60.0]

if True:
    x_1, y_1 = x[250:300], y[250:300]
    pos_1 = 21.4
    fit_1, r_squared1 = gauss_fit(x_1, y_1, pos_1)


    x_2, y_2 = x[1080:1140], y[1080:1140]
    pos_2 = 84.3
    fit_2, r_squared2 = gauss_fit(x_2, y_2, pos_2)
    
    plt.plot(x, y, linewidth = 0.4, label="spectre du cadmium")
    plt.plot(x, gaussienne(x, *fit_1), linewidth = 0.9, 
            label=f"$Curve$ $fit$ d'une gaussienne à {round(fit_1[2], sigfigs=3)} KeV\n$\sigma$={round(fit_1[1], sigfigs=3)}\n$R^2$ = {round(r_squared1, sigfigs=3)}")
    plt.plot(x, gaussienne(x, *fit_2), linewidth = 0.9, 
            label=f"$Curve$ $fit$ d'une gaussienne à {round(fit_2[2], sigfigs=3)} KeV\n$\sigma$={round(fit_2[1], sigfigs=3)}\n$R^2$ = {round(r_squared2, sigfigs=3)}")
    plt.legend()
    plt.show()
    
res3 = [0.7896, 1.48]
pics3 = [22.8, 84.7]

res1 = [0.717, 0.708, 2.181]
pics1 = [7.03, 15.0, 122.0]

plt.scatter(pics1, np.array(res1)/np.array(pics1)*100, label='Pics du Cobalt')
plt.scatter(pics2, np.array(res2)/np.array(pics2)*100, label="Pics de l'Américium")
plt.scatter(pics3, np.array(res3)/np.array(pics3)*100, label='Pics du Cadmium')
plt.legend()
plt.xlabel('Énergie [KeV]')
plt.ylabel('Résolution [%]')
plt.show()