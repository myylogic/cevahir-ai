import numpy as np, zsim
from zsim import *
cfg=Cfg(); eng=Engine(cfg, ncap=40000); world=World(seed=1)
g=world.stream(6000)
inv=None
for i in range(1,5001):
    eng.moment(next(g))
inv={u:k for k,u in eng.sym.items()}
def nm(u): return inv.get(int(u), "comp%s"%(tuple(inv.get(p,'?') for p in eng.parts.get(int(u),())),))
# trace 3 moments in detail: re-implement settle verbosely
for j in range(3):
    syms=next(g)
    # peek: what are the pending intero + ext
    ext=[eng.unit(x) for x in syms]
    pf_ids,pf_vals=eng.prev_focus
    print("\n=== moment", eng.t+1, "ext", syms, "pending intero", [nm(u) for u in eng.pending])
    print("  prev focus:", [(nm(u), round(float(a),2)) for u,a in zip(pf_ids,pf_vals)])
    # net input to each ext unit from the prev focus (context) at iteration 1
    for x in ext:
        tot=0; parts=[]
        for u,a in zip(pf_ids.tolist(), (pf_vals*cfg.decay).tolist()):
            A=eng.out[0][u]
            if A is None: continue
            jx=A.idx.get(x)
            if jx is None: continue
            d,w=eng.weights(u,0)
            tot+=a*w[jx]; parts.append((nm(u), round(float(w[jx]),2)))
        print("  ctx->", nm(x), "sum=%.2f"%tot, parts, " => net at it1 = %.2f"%(1+tot))
    ids,vals=eng.moment(syms)
    print("  focus:", [(nm(u), round(float(a),2)) for u,a in zip(ids,vals)])
