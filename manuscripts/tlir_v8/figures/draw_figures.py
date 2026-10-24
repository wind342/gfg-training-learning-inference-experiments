"""Reproducible publication figures for six-system TLI evidence.
All evidence indicators reflect frozen source protocols/results; no simulated measurements.
Source keys in README and manuscript. Statuses are independent within each executed network.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from matplotlib import rcParams

out = Path(__file__).resolve().parent
rcParams.update({
    'font.family':'DejaVu Sans', 'svg.fonttype':'none', 'pdf.fonttype':42,
    'font.size':8.0,
    'savefig.facecolor':'white',
})
navy='#132D43'; blue='#27799A'; deep='#167A79'; mint='#E4F2EF'; pale='#EEF5FA'; gray='#5E6E7A'; line='#B3C6CC'; amber='#B6731E'; amber_bg='#FFF1D7'; light='#E9ECEF'; purple='#5965A5'

def panel(ax,x,y,w,h,fill,title,desc,foot):
    shape=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.035',facecolor=fill,edgecolor='#B6C9CF',linewidth=.72,transform=ax.transAxes)
    ax.add_patch(shape)
    ax.text(x+w/2,y+h-.14,title,transform=ax.transAxes,ha='center',va='top',fontsize=8.05,color=navy,weight='bold')
    ax.text(x+w/2,y+h*.49,desc,transform=ax.transAxes,ha='center',va='center',fontsize=7.15,color=navy,linespacing=1.18)
    ax.text(x+w/2,y+.11,foot,transform=ax.transAxes,ha='center',va='center',fontsize=6.4,color=gray)

def arrow(ax,p1,p2,style='-|>',color=blue,lw=1.15,mutation=9,rad=0,linestyle='-'):
    a=FancyArrowPatch(p1,p2,arrowstyle=style,mutation_scale=mutation,linewidth=lw,color=color,linestyle=linestyle,connectionstyle=f'arc3,rad={rad}',transform=ax.transAxes)
    ax.add_patch(a)

fig,ax=plt.subplots(figsize=(7.2,2.60))
fig.subplots_adjust(left=.013,right=.987,top=.98,bottom=.03)
ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.5,.971,'FOUR ROLES OF UNIFIED TLIR DYNAMICS',ha='center',va='top',fontsize=11.0,weight='bold',color=navy,transform=ax.transAxes)
ax.text(.5,.888,'TLI tested across six systems  •  Feedback intervention tested in GRU (RL-E05/E06)',ha='center',va='top',fontsize=7.2,color=gray,transform=ax.transAxes)
xs=[.019,.271,.523,.775]; y=.305; w=.207; h=.49
roles=[
    ('TRAINING','Actual action on\nhistorical receiving state','update + state swaps'),
    ('LEARNING','Persistent support\nformation and readout','component gates + boundaries'),
    ('INFERENCE','Frozen, query-conditioned\nrecruitment of learned support','version rollback'),
    ('REINFORCEMENT','Correct feedback changes\nfuture functional formation','feedback dose + recovery')]
cols=[pale,mint,pale,mint]
for i,(name,desc,foot) in enumerate(roles): panel(ax,xs[i],y,w,h,cols[i],name,desc,foot)
for i in range(3):arrow(ax,(xs[i]+w+.004,y+h/2),(xs[i+1]-.008,y+h/2))
# The feedback loop returns to Training without demoting Reinforcement to an ancillary theory role.
arrow(ax,(xs[3]+w-.045,y-.015),(.892,.125),color=purple,lw=1.1,linestyle='--')
arrow(ax,(.891,.120),(.124,.120),color=purple,lw=1.1,linestyle='--')
arrow(ax,(.124,.125),(.124,y-.016),color=purple,lw=1.1,linestyle='--')
ax.text(.507,.159,'FEEDBACK  →  NEXT TRAINING ACTION',fontsize=7.6,color=purple,ha='center',transform=ax.transAxes)
for ext in ('pdf','svg','png'):
    fig.savefig(out/f'figure1_functional_chain.{ext}',dpi=300,bbox_inches='tight',pad_inches=.025)
plt.close(fig)

fig,ax=plt.subplots(figsize=(7.2,4.0))
fig.subplots_adjust(left=.025,right=.98,top=.965,bottom=.055)
ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.5,.984,'COMPLETE TRAINING–LEARNING–INFERENCE TESTS BY SYSTEM',ha='center',va='top',fontsize=9.8,weight='bold',color=navy,transform=ax.transAxes)
ax.text(.5,.938,'Each row links actual training updates to inference on the same learned version',ha='center',va='top',fontsize=7.5,color=gray,transform=ax.transAxes)
left=.012; start=.195; dx=.132; gap=.014; cols=['REALIZED\nACTION','FINITE\nRESPONSE','SUPPORT\nREORG.','TARGET\nBOUNDARY','FROZEN\nINFERENCE']
for j,c in enumerate(cols):
    ax.text(start+j*(dx+gap)+dx/2,.844,c,ha='center',va='center',fontsize=7.0,weight='bold',color=navy,transform=ax.transAxes,linespacing=1.12)
rows=[
    ('nanoGPT','Transformer / Adam', ['pass']*5, '13 histories; exact versions'),
    ('ResNet-18','SGD momentum', ['pass']*5, 'TL-G01 + INF-G01'),
    ('Diffusion U-Net','AdamW', ['pass']*5, 'TL-G02 + INF-G01'),
    ('Vision-MoE','Top-2 routing', ['pass']*5, 'TL-G03 + INF-G02'),
    ('GNN','RMSprop', ['pass','mixed','pass','pass','pass'], '2/3 nonlinear-rate gates'),
    ('Hard-spiking SNN','Adagrad', ['pass']*5, '4/4 full-chain gates'),
]
ys=[.752,.652,.552,.452,.352,.252]
for i,(name,kind,stat,short) in enumerate(rows):
    yy=ys[i]
    ax.text(.016,yy+.028,name,ha='left',va='center',fontsize=8.15,weight='bold',color=navy,transform=ax.transAxes)
    ax.text(.016,yy-.015,kind,ha='left',va='center',fontsize=6.85,color=blue,transform=ax.transAxes)
    for j,val in enumerate(stat):
        x=start+j*(dx+gap)
        bg=mint if val=='pass' else amber_bg
        fg=deep if val=='pass' else amber
        ax.add_patch(FancyBboxPatch((x,yy-.011),dx,.062,boxstyle='round,pad=.004,rounding_size=.013',transform=ax.transAxes,facecolor=bg,edgecolor='#CBD9DC' if val=='pass' else '#E8D4AE',linewidth=.55))
        ax.text(x+dx/2,yy+.021,'PASS' if val=='pass' else '2/3 SEEDS',fontsize=7.15,ha='center',va='center',color=fg,weight='bold',transform=ax.transAxes)
        if j<4: arrow(ax,(x+dx+.004,yy+.02),(x+dx+gap-.004,yy+.02),color=line,lw=.65,mutation=7)
    # line at bottom separators
    if i<5: ax.plot([.01,.996],[yy-.045,yy-.045],color='#E8EDF0',lw=.55,transform=ax.transAxes)

ax.text(.023,.148,'FROZEN OUTCOME',fontsize=7.1,weight='bold',color=navy,transform=ax.transAxes)
ax.text(.205,.148,'Transformer, ResNet, diffusion, and MoE: all registered role-level criteria supported',fontsize=7.05,color=navy,transform=ax.transAxes)
ax.text(.205,.101,'GNN: one seed failed the nonlinear-frequency threshold; other eight gates passed in all seeds',fontsize=7.05,color=navy,transform=ax.transAxes)
ax.text(.205,.056,'SNN: 12/12 frozen relationship / integrity gates passed in each of four seeds',fontsize=7.05,color=navy,transform=ax.transAxes)
for ext in ('pdf','svg','png'):
    fig.savefig(out/f'figure2_six_system_evidence.{ext}',dpi=300,bbox_inches='tight',pad_inches=.025)
plt.close(fig)
print('Created figures:',*[p.name for p in out.glob('figure*.pdf')],sep='\n  ')
