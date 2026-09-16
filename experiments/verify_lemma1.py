import numpy as np
rng=np.random.default_rng(0)
a,b=0.05,0.10; lam=1-a-b; pi=np.array([b/(a+b),a/(a+b)])
e=np.array([0.05,0.40]); ebar=pi@e; var_e=pi@(e**2)-ebar**2
phi_pred=1+2*lam/(1-lam)*var_e/(ebar*(1-ebar))
T=50; R=200000
A=np.array([[1-a,a],[b,1-b]])
s=rng.choice(2,size=R,p=pi); N=np.zeros(R); allc=np.ones(R,bool)
for t in range(T):
    E=rng.random(R)<e[s]; N+=E; allc&=~E
    s=np.where(rng.random(R)<A[s,1],1,0)
# finite-T prediction from (3.5)
k=np.arange(1,T); varT=T*ebar*(1-ebar)+2*var_e*np.sum((T-k)*lam**k)
print(f"mean N: sim {N.mean():.3f}  pred {T*ebar:.3f}")
print(f"var  N: sim {N.var():.3f}  pred(3.5) {varT:.3f}   binomial {T*ebar*(1-ebar):.3f}")
print(f"phi asymptotic pred {phi_pred:.3f}; finite-T ratio {varT/(T*ebar*(1-ebar)):.3f}")
print(f"P(all correct): sim {allc.mean():.4f}  indep (1-ebar)^T {(1-ebar)**T:.4f}")
