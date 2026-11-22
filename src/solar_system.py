import numpy as np
import matplotlib.pyplot as plt

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
M=np.diag(m[:,0])

#positions initiales
q_0=np.array([[0,0,0], [-3.5023653, -3.8169847, -1.5507963],[9.0755314,-3.0458353,-1.6483708],
              [8.3101420,-16.2901086,-7.2521278], [11.4707666,-25.7294829,-10.8169456],
               [-15.5387357, -25.2225594, -3.1902382]]) # vecteur position inital
v_0=np.array([[0, 0, 0], [0.00565429,-0.00412490,-0.00190589], [0.00168318,0.00483525,0.00192462],
              [0.00354178,0.00137102,0.00055029], [0.00288930,0.00114527,0.00039677],
              [0.00276725,-0.00170702,-0.00136504]])
p_0=m*v_0 # vecteur quantité de mouvement initial

#Changement de référentiel pour se mettre dans celui du barycentre
e=np.ones(len(m))
u=e@p_0/np.dot(e,m)

q_0 -= e@(m*q_0)/np.dot(e,m) 
p_0 -= m*u 


def H(q,p):
    """ Renvoie le hamiltonien du problème """
    n=len(p)
    H=0.0
    for i in range(n):
        H+=1/2*np.linalg.norm(p[i])**2/m[i,0]
        s=0.0
        for j in range(i):
            s+= - G*m[i,0]*m[j,0]/np.linalg.norm(q[i]-q[j])
        H+=s
    return H

def grad_U(q):
    t=np.zeros((len(q), 3))
    for i in range(len(q)):
        u=0
        for j in range(len(q)):
            if j==i:
                continue
            u+= G*m[i]*m[j]*(q[i]-q[j])/np.linalg.norm(q[i]-q[j])**3
        t[i]=u
    return t

def P(p):
    """ Renvoie P=somme des pi"""
    return p.sum(axis=0)

def L(q,p):
    """ Moment cinétique total"""
    return np.cross(q,p).sum(axis=0)

def f(z):
    q, p = z
    return np.array([p/m, -grad_U(q)])

## Euler explicite

def step_Euler_explicite(q,p,h):
    z = np.array([q, p])
    return z+h*f(z)

## RK4

def step_RK4(q, p, h):
    z = np.array([q, p])
    k1=f(z)
    k2=f(z+h/2*k1)
    k3=f(z+h/2*k2)
    k4=f(z+h*k3)
    return z+h/6*(k1+2*k2+2*k3+k4)

## Euler symplectique

def step_Euler_symplectique(q,p,h):
    p=p-h*grad_U(q)
    q=q+h*p/m
    return (q,p)

## Stormer-Verlet

def step_Stormer_Verlet(q, p, h, u):
    q=q+h*(p-h/2*u)/m
    v=grad_U(q)
    p=p-h/2*(u+v)
    return ((q,p), v)



def integrate(step, q0, p0, h, N):
    z_stock=[(q0, p0)]
    z=(q0, p0)
    if step ==step_Stormer_Verlet:
        u=grad_U(q0) ## permet d'optimiser le calcul de stormer Verlet en réutilisant grad_u(q_(n+1)) à l'étape suivante
    for _ in range(N):
        if step ==step_Stormer_Verlet:
            z, u=step(z[0], z[1], h, u)
        else:
            z=step(z[0], z[1], h)
        z_stock.append(z)
    Z = np.array(z_stock)   
    return Z[:, 0], Z[:, 1]

## fonctions de plot
noms = ["Soleil", "Jupiter", "Saturne", "Uranus", "Neptune", "Pluton"]

def plot_orbites(Q, titre):
    for i in range(6):
        plt.plot(Q[:, i, 0], Q[:, i, 1], label=noms[i])
    plt.axis("equal")
    plt.title(titre)
    plt.legend()

def plot_energie(Qs, Ps, h, titre):
    E = np.array([H(Qs[k], Ps[k]) for k in range(len(Qs))])
    t = h*np.arange(len(Qs))
    plt.semilogy(t, np.abs((E - E[0])/E[0]), label=titre)
    plt.xlabel("t (jours)")
    plt.ylabel("erreur relative sur H")

def erreurs_invariants(Qs, Ps):
    """Renvoie l'erreur absolue sur P et l'erreur relative sur L à chaque pas."""
    Pt = np.array([P(Ps[k]) for k in range(len(Qs))])
    Lt = np.array([L(Qs[k], Ps[k]) for k in range(len(Qs))])
    err_P = np.linalg.norm(Pt - Pt[0], axis=1)
    err_L = np.linalg.norm(Lt - Lt[0], axis=1) / np.linalg.norm(Lt[0])
    return err_P, err_L


def plot_invariants(resultats, h, pas_affichage=20):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (nom, (Qs, Ps)) in zip(axes.flat, resultats.items()):
        err_P, err_L = erreurs_invariants(Qs, Ps)
        print(f"{nom:20s} max |P - P0| = {err_P.max():.2e}   max err. rel. L = {err_L.max():.2e}")

        t = h*np.arange(len(Qs))
        k = slice(1, None, pas_affichage)          # on saute t=0 (erreur nulle)
        ax.semilogy(t[k], np.maximum(err_L[k], 1e-17))
        ax.axhline(1e-16, color="gray", ls="--", lw=0.8)   # précision machine
        ax.set_title(nom)
        ax.set_ylabel("erreur relative sur L")
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("t (jours)")
    fig.suptitle("Conservation du moment cinétique")
    fig.tight_layout()

def main():
    h = 100        # pas de temps en jours. Prendre plutôt 100 ou 200 pour voir grosse divergence de RK4, Euler explicite
    N = 20000     # nombre de pas, donc nombre de jours = h*N

    methodes = {
        f"Euler explicite avec h={h}": step_Euler_explicite,
        f"Euler symplectique avec h={h}": step_Euler_symplectique,
        f"RK4 avec h={h}": step_RK4,
        f"Störmer-Verlet avec h={h}": step_Stormer_Verlet,
    }

    # Intégration
    resultats = {}
    for nom, step in methodes.items():
        print(f"Intégration : {nom}")
        resultats[nom] = integrate(step, q_0.copy(), p_0.copy(), h, N)

    # 1) Orbites 
    plt.figure(figsize=(12, 10))
    for k, (nom, (Qs, Ps)) in enumerate(resultats.items()):
        plt.subplot(2, 2, k + 1)
        plot_orbites(Qs, nom)
    plt.tight_layout()

    # 2) Erreur sur l'énergie
    plt.figure()
    for nom, (Qs, Ps) in resultats.items():
        plot_energie(Qs, Ps, h, nom)
    plt.legend()
    plt.title("Erreur relative sur le hamiltonien")

    # 3) Invariants P et L
    plot_invariants(resultats, h)

    plt.show()


if __name__ == "__main__":
    main()