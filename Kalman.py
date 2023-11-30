# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:12:13 2023

@author: bounn
"""

import numpy as np
import matplotlib.pyplot as plt



##création de la fonction 

def kalman(obs, u, proc_cov_matrice, mesure_cov_matrice,A,B,C):
    
    ##Initialisation
    X_t=np.zeros((2,obs.shape[1])) ##position théorique
    X_t[:,0]=obs[:,0]
    X=np.zeros((2,obs.shape[1]-1)) ##état actuel
    liste_P_est=[proc_cov_matrice]
    liste_P=[proc_cov_matrice]
    
    for i in np.arange(1,obs.shape[1]):
        
        ##Premiere étape mise a jour de la position théoriquement
        X_t[:,i]=np.dot(A,X_t[:,i-1])+np.dot(B,u) 
    
        ##estimation matrice de covariance du processus 
        P_est= np.dot(np.dot(A,liste_P[i-1]),np.transpose(A))+ proc_cov_matrice
        liste_P_est.append(P_est)
        
        ##Troisième étape Kalman Gain
        CEC=np.dot(np.dot(C,P_est),np.transpose(C))
        EC=np.dot(P_est,np.transpose(C))
        inv=np.linalg.inv(CEC+mesure_cov_matrice)
        K=np.dot(EC,inv)
        
        
        ##4eme étape état actuel (ésperance)
        X[:,i-1]=np.transpose(X_t[:,i]) + np.transpose(np.dot(K,obs[:,i]-np.transpose(np.dot(C,X_t[:,i]))))
 
        ##5eme étape mise a jour de la matrice de covariance du processus
        KC=np.dot(K,C)
        P=np.dot(np.eye(len(proc_cov_matrice))-KC,P_est)
        liste_P.append(P)
        
    return  X,X_t[:,1:obs.shape[1]], liste_P[1:obs.shape[1]]                     

##Application  

#On dispose d'observations passées sur la position et la vitesse d'un objet se déplacant en ligne droite et on souhaite connaitre sa position actuelle.
#On a une vitesse initialle de 50m/s, une position initiale à 50m et on suppose que l'accélération vaut 1m/s²
#modèle : Xt[n]=AXt[n-1]+ Bu + wk  avec wk~N(0,mesure_cov_matrice)
#         X[n]=CXt[n] + ek         avec ek~N(0,proc_cov_matrice) 
#On souhaite appliquer le filtre de Kalman pour avoir les positions et les vitesses  (c'est un vecteur d'ésperance ), ainsi que la matrice de var cov.
    
 
dt=1 
A=np.array([[1,dt],[0,1]])
B=np.array([dt**2/2,dt]) 
C=np.diag([1,1])
obs=np.array([[50,98,140,197],[50,52,52.7,54]]) ## les observations, premier vecteur position, deuxieme vecteur vitesse

mesure_cov_matrice=np.diag([1,1]) ##on suppose une erreur de mesure de 1m pour la position et 1m/s la vitesse
proc_cov_matrice=np.diag([3,1]) ##on suppose une erreur exterieure sur la position de 3m et 1 m/s pour la vitesse
u=1 ##on suppose une acceleration (u) de 1 m/s² 


Kal=kalman(obs, u,proc_cov_matrice, mesure_cov_matrice,A,B,C)

matrices=Kal[2] ##les matrices de var/cov associées aux vecteurs d'esperance

absc=np.arange(1,4)

plt.subplot(1,2,1)

##Plot des positions 
plt.plot(absc,Kal[0][0,:],'r.', ) ##état calculé
plt.plot(absc,Kal[1][0,:],'k*') ##valeur théorique 
plt.plot(absc,obs[0,1:4],'b*')  ##observations
plt.title("Positions")
plt.legend([ 'état calculé',"valeur théorique","observations"] ,fontsize =10)
plt.xlabel('t',fontsize=10)
plt.ylabel('position',fontsize=10)

plt.subplot(1,2,2)
##Plot des vitesses
plt.plot(absc,Kal[0][1,:],'r.') ##état calculé
plt.plot(absc,Kal[1][1,:],'k*') ##valeur théorique 
plt.plot(absc,obs[1,1:4],'b*')
plt.title("Vitesses")
plt.legend([ 'état calculé',"valeur théorique","observations"] ,fontsize =10)
plt.xlabel('t',fontsize=10)
plt.ylabel('vitesse',fontsize=10)
plt.tight_layout()







