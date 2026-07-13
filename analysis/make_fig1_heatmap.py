"""Regenerate Fig. 1 (ceiling-fit R^2 heatmap, full ten-level grid, corrected data)."""
import json, numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
import subprocess, csv
D=json.load(open(os.path.join(ROOT,'data/campaign_main_mc50_fixed.json')))['results']
NS=np.array([5,10,15,20,25,30,40,50,75,100,150,200],float)
TRS=[0.05,0.10,0.13,0.15,0.17,0.20,0.25,0.30,0.35,0.40]
X=np.column_stack([np.ones_like(NS),1/np.sqrt(NS)])
M=[]
for tj in ['circle','square','lemniscate','zigzag']:
    row=[]
    for tr in TRS:
        y=np.array([D[f"{tj}_tr{tr:.2f}_N{int(N)}_T0"]['rmse_mean'] for N in NS])
        b,_,_,_=np.linalg.lstsq(X,y,rcond=None)
        row.append(1-np.sum((y-X@b)**2)/np.sum((y-y.mean())**2))
    M.append(row)
M=np.array(M)
fig,ax=plt.subplots(figsize=(17.7,9.1),dpi=300)
im=ax.imshow(M,cmap='RdYlGn',vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(len(TRS))); ax.set_xticklabels([f"{int(t*100)}%" for t in TRS],fontsize=16)
ax.set_yticks(range(4)); ax.set_yticklabels(['Circle','Square','Lemniscate','Zigzag'],fontsize=16)
ax.set_xlabel('Adversarial ratio tr',fontsize=18); ax.set_ylabel('Trajectory',fontsize=18)
ax.set_title('Ceiling fit R² (RMSE(N) = a + b/√N), baseline adversary T0, MC = 50',fontsize=18)
for i in range(4):
    for j in range(len(TRS)):
        ax.text(j,i,f"{M[i,j]:.2f}",ha='center',va='center',fontsize=14)
cb=fig.colorbar(im,ax=ax,fraction=0.03,pad=0.02); cb.set_label('R²',fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(ROOT,'figures/fig1_heatmap.png'),dpi=300,bbox_inches='tight')
plt.savefig(os.path.join(ROOT,'figures/fig1_heatmap.pdf'),bbox_inches='tight')
print("figures/fig1_heatmap.{png,pdf} regenerated")
