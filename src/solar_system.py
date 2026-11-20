import numpy as np

## Paramètres du système 

G=2.95912208286e-4

# masses
m_0=1.00000597682 # Soleil
m_1=0.000954786104043 # Jupiter
m_2=0.000285583733151 # Saturne
m_3=0.0000437273164546 # Uranus
m_4=0.0000517759138449 # Neptune
m_5=1/(1.3e8) # Pluton
m=np.array([m_0, m_1, m_2, m_3, m_4, m_5])[:, None]
M=np.diag(m)

#positions initiales
q_0=np.array([[0,0,0], [-3.5023653, -3.8169847, -1.5507963],[9.0755314,-3.0458353,-1.6483708],
              [8.3101420,-16.2901086,-7.2521278], [11.4707666,-25.7294829,-10.8169456],
               [-15.5387357, -25.2225594, -3.1902382]])
v_0=np.array([[0, 0, 0], [0.00565429,-0.00412490,-0.00190589], [0.00168318,0.00483525,0.00192462],
              [0.00354178,0.00137102,0.00055029], [0.00288930,0.00114527,0.00039677],
              [0.00276725,-0.00170702,-0.00136504]])
p_0=m*v_0

#Changement de référentiel pour se mettre dans celui du barycentre
e=np.ones(len(m))
u=e@p_0/np.dot(e,m)

q_0 -= e@(m*q_0)/np.dot(e,m)
p_0 -= m*u


def H(p,q):
    return 1/2*p.T@np.linalg.inv(M)@p -G

