#!/usr/bin/env python3
"""
make_figures_revision.py — main-text Figures 1–4 and Supplementary Figures S1–S2 (revision).

Layout (unchanged table numbering; 4 figures + 4 tables):
  Fig 1  four reference paths + realised agent paths at N = 5 and N = 200 (new; R1 item 3)
  Fig 2  (a) crowd coefficient b vs adversarial ratio, 95% CI (new; R1 item 6)
         (b) ceiling-fit R² heat map, six-level grid as Table 1, corrected data (old Fig 1)
  Fig 3  RMSE vs N for tr = 5/20/40%, N = 5–200 with the a + b/√N fit extended to 3,200 and the
         measured floor at N = 400–3,200 (old Figs 2 + 3 merged)
  Fig 4  adversary ladder T0/T1/T2 at tr = 40% (old Fig 4, corrected zigzag)
  S1     two-way ANOVA η² with exact P values (P computed here; underflow shown as < 10⁻³⁰⁰)
  S2     pipeline flowchart (schematic; run counts from the data files)
  S3     size axis a(R) with the directly measured vote-noise floor
  S4     shape axis: floor vs n-gon sides and the failed Λ collapse
  S5     phase-averaged error localisation index

READS : ../data/campaign_main_mc50_fixed_zzfix.json, largeN_saturation.json, anova_raw_mc50.json,
        localization_index_ci.json (caption numbers), engine for Fig 1 traces
WRITES: ../figures/fig1_trajectories.{png,pdf} … figS2_pipeline.{png,pdf}
RUN   : cd code ; python make_figures_revision.py            (all)
        python make_figures_revision.py --fig 3               (one figure: 1,2,3,4,S1,S2)
Spec  : Sci Rep — sans-serif 8 pt, lines ≥ 1 pt, 2-column 180 mm, panel labels bold a b c d,
        PNG 600 dpi + PDF vector.
"""
import sys, os, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import adversary_ladder as E
from zigzag_periodic import PeriodicZigzag

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data'); FIG = os.path.join(ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)
MM = 1/25.4; TWO = 180*MM; ONE = 88*MM
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'lines.linewidth': 1.2,
                     'axes.linewidth': 1.0, 'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
                     'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
GEOMS = ['circle', 'square', 'lemniscate', 'zigzag']
LABEL = {'circle': 'Circle', 'square': 'Square', 'lemniscate': 'Lemniscate', 'zigzag': 'Zigzag'}
COL = {'circle': '#1f77b4', 'square': '#2ca02c', 'lemniscate': '#9467bd', 'zigzag': '#d62728'}
NS = np.array([5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200], float)
TRS = [0.05, 0.10, 0.13, 0.15, 0.17, 0.20, 0.25, 0.30, 0.35, 0.40]
TR6 = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
X = np.column_stack([np.ones_like(NS), 1/np.sqrt(NS)])

def load(n): return json.load(open(os.path.join(DATA, n)))
def save(fig, name):
    fig.savefig(os.path.join(FIG, name+'.png'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(FIG, name+'.pdf'), bbox_inches='tight'); plt.close(fig)
    print(f"  figures/{name}.{{png,pdf}}")
def panel(ax, s, x=-0.14, y=1.04): ax.text(x, y, s, transform=ax.transAxes, fontweight='bold', fontsize=9, va='bottom')

R = load('campaign_main_mc50_fixed_zzfix.json')['results']
def cell(g, tr, N, m='T0'): return R[f"{g}_tr{tr:.2f}_N{int(N)}_{m}"]
def curve(g, tr, m='T0'):
    y = np.array([cell(g, tr, N, m)['rmse_mean'] for N in NS]); ci = np.array([cell(g, tr, N, m)['rmse_ci95'] for N in NS])
    sd = np.array([cell(g, tr, N, m)['rmse_std'] for N in NS]); return y, ci, sd
def ols(y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; r2 = 1-np.sum((y-X@b)**2)/np.sum((y-y.mean())**2); return b, r2
def wls(y, sd, mc=50):
    w = mc/sd**2; W = np.diag(w); C = np.linalg.inv(X.T@W@X); b = C@X.T@W@y
    return b, np.sqrt(np.diag(C))

# ------------------------------------------------------------------ Fig 1
def trace(tj, N, tr, seed):
    rng = np.random.default_rng(seed); pos = tj.start(); vel = np.zeros(2); h = [pos.copy()]
    pang = 0.; cd = np.array([1., 0.]); err = []
    for f in range(E.FRAMES):
        if f % E.VOTE_INT == 0:
            di = max(0, len(h)-1-E.DELAY_F); dp = h[di]; _, a0 = tj.closest(dp); lap = tj.at(a0+E.LOOK)
            d = lap-dp; n = np.linalg.norm(d)
            if n > 1e-10: d /= n
            ia = np.degrees(np.arctan2(d[1], d[0])); v = E.gen_votes_adv(ia, pang, tr, N, rng, 'T1', 0.0, None); pang = ia
            bl = E.DIRS[v].mean(axis=0); g = np.linalg.norm(bl); cd = bl/g if g > 1e-10 else np.array([1., 0.])
        vel += E.SMOOTH*(cd*5.0-vel); pos = pos+vel*E.DT; h.append(pos.copy()); cp, _ = tj.closest(pos); err.append(np.linalg.norm(pos-cp))
    return np.array(h), float(np.sqrt(np.mean(np.array(err)**2)))

def fig1():
    tr = 0.20
    objs = {'circle': E.Circle(), 'square': E.Square(), 'lemniscate': E.Lemniscate(), 'zigzag': PeriodicZigzag()}
    fig = plt.figure(figsize=(TWO, TWO*0.62))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 0.62], hspace=0.45, wspace=0.35)
    axes = {'circle': fig.add_subplot(gs[0, 0]), 'square': fig.add_subplot(gs[0, 1]), 'lemniscate': fig.add_subplot(gs[0, 2]), 'zigzag': fig.add_subplot(gs[1, :])}
    for k, g in enumerate(GEOMS):
        ax = axes[g]; tj = objs[g]
        if g == 'zigzag':
            xs = np.linspace(0, 60, 1200); ref = np.array([tj.at(x*tj.L/tj.sx) for x in xs])
        elif g == 'circle':
            t = np.linspace(0, 2*np.pi, 400); ref = np.column_stack([10*np.cos(t), 10*np.sin(t)])
        elif g == 'square': ref = tj.c
        else: ref = np.vstack([tj.pts, tj.pts[:1]])
        ax.plot(ref[:, 0], ref[:, 1], color='0.55', lw=2.0, label='reference path', zorder=1)
        rms = {}
        for N, c, z in [(5, '#e6821e', 2), (200, '#1f77b4', 3)]:
            h, rm = trace(tj, N, tr, seed=int(np.random.SeedSequence([2034, k, N, int(tr*1000)]).generate_state(1)[0]))
            if g == 'zigzag': h = h[h[:, 0] <= 60]
            ax.plot(h[:, 0], h[:, 1], color=c, lw=1.0, alpha=0.85, label=f'crowd of {N}', zorder=z); rms[N] = rm
        ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_title(f"{LABEL[g]}\nRMSE {rms[5]:.2f} m (N = 5) · {rms[200]:.2f} m (N = 200)" if g != 'zigzag' else f"{LABEL[g]}   RMSE {rms[5]:.2f} m (N = 5) · {rms[200]:.2f} m (N = 200)", pad=3, fontsize=7.5)
        if g in ('circle', 'square'): ax.set_xlim(-13.5, 13.5); ax.set_ylim(-13.5, 13.5)
        elif g == 'lemniscate': ax.set_xlim(-9.5, 9.5); ax.set_ylim(-9.5, 9.5)
        else: ax.set_ylim(-3.5, 8.5)
        if g == 'zigzag': ax.legend(loc='upper right', fontsize=7, ncol=3, handlelength=1.6, bbox_to_anchor=(1.0, 1.02))
        panel(ax, 'abcd'[k], x=-0.22 if g != 'zigzag' else -0.06, y=1.02 if g != 'zigzag' else 1.04)
    save(fig, 'fig1_trajectories')

# ------------------------------------------------------------------ Fig 2
def fig2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(TWO, TWO*0.40), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.35))
    for g in GEOMS:
        bs, se = [], []
        for tr in TRS:
            y, ci, sd = curve(g, tr); p, s = wls(y, sd); bs.append(p[1]); se.append(s[1])
        bs = np.array(bs); se = np.array(se); trs = np.array(TRS)
        a.errorbar(trs*100, bs, yerr=1.96*se, fmt='o-', ms=3.5, color=COL[g], capsize=2, lw=1.2, label=LABEL[g])
        sl, ic, r, _, _ = stats.linregress(trs, bs)
        a.plot(trs*100, ic+sl*trs, ls='--', lw=1.0, color=COL[g], alpha=0.6)
    a.axhline(0, color='0.5', lw=1.0); a.set_xlabel('Adversarial ratio tr (%)'); a.set_ylabel('Crowd coefficient b (m)')
    a.legend(loc='upper left'); panel(a, 'a')
    M = np.array([[ols(curve(g, tr)[0])[1] for tr in TR6] for g in GEOMS])
    im = b.imshow(M, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    b.set_xticks(range(6)); b.set_xticklabels([f"{int(t*100)}%" for t in TR6]); b.set_yticks(range(4)); b.set_yticklabels([LABEL[g] for g in GEOMS])
    b.set_xlabel('Adversarial ratio tr'); b.spines[['top', 'right']].set_visible(True)
    for i in range(4):
        for j in range(6): b.text(j, i, f"{M[i, j]:.2f}", ha='center', va='center', fontsize=7.5, color='black' if 0.25 < M[i, j] < 0.85 else 'white')
    cb = fig.colorbar(im, ax=b, fraction=0.045, pad=0.03); cb.set_label('Ceiling-fit R²'); cb.outline.set_linewidth(1.0)
    panel(b, 'b', x=-0.30)
    save(fig, 'fig2_b_vs_tr_and_heatmap'); return M

# ------------------------------------------------------------------ Fig 3
def fig3():
    L = load('largeN_saturation.json')['results']; BIG = [400, 800, 1600, 3200]
    fig, axes = plt.subplots(2, 2, figsize=(TWO, TWO*0.72), sharex=True); axes = axes.ravel()
    Nx = np.logspace(np.log10(5), np.log10(3200), 200)
    for k, g in enumerate(GEOMS):
        ax = axes[k]
        for tr, c, mk in [(0.05, '#4c9ed9', '^'), (0.20, '#e6821e', 'o'), (0.40, '#c0508a', 's')]:
            y, ci, sd = curve(g, tr); p, _ = ols(y)
            ax.errorbar(NS, y, yerr=ci, fmt=mk+'-', ms=3.5, color=c, capsize=1.5, lw=1.2, label=f'tr = {int(tr*100)}%')
            ax.plot(Nx, p[0]+p[1]/np.sqrt(Nx), ls='--', lw=1.0, color=c, alpha=0.7)
            yb = [L[f"{g}_tr{tr:.2f}_N{N}"]['rmse_mean'] for N in BIG]; cb = [L[f"{g}_tr{tr:.2f}_N{N}"]['rmse_ci95'] for N in BIG]
            ax.errorbar(BIG, yb, yerr=cb, fmt=mk, ms=4.0, mfc='white', mec=c, color=c, capsize=1.5, lw=1.0)
        ax.set_xscale('log'); ax.set_xlim(4, 4000); ax.set_xticks([5, 20, 50, 200, 800, 3200]); ax.set_xticklabels(['5', '20', '50', '200', '800', '3200'])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.axvspan(300, 3800, color='0.93', zorder=0); ax.set_title(LABEL[g], pad=3); panel(ax, 'abcd'[k])
        if k in (2, 3): ax.set_xlabel('Crowd size N')
        if k in (0, 2): ax.set_ylabel('RMSE (m)')
        if k == 0: ax.legend(loc='upper right')
    fig.subplots_adjust(hspace=0.28, wspace=0.22); save(fig, 'fig3_rmse_vs_crowd_size')

# ------------------------------------------------------------------ Fig 4
def fig4():
    fig, axes = plt.subplots(1, 4, figsize=(TWO, TWO*0.34)); tr = 0.40
    for k, g in enumerate(GEOMS):
        ax = axes[k]
        for m, c, lab in [('T0', '#1f77b4', 'T0 uniform'), ('T1', '#e6821e', 'T1 committed'), ('T2', '#7f7f7f', 'T2 persistent')]:
            y, ci, sd = curve(g, tr, m); ax.errorbar(NS, y, yerr=ci, fmt='o-', ms=3, color=c, capsize=1.5, lw=1.2, label=lab)
        ax.set_xscale('log'); ax.set_xlim(4, 260); ax.set_xticks([5, 20, 50, 200]); ax.set_xticklabels(['5', '20', '50', '200'])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.set_title(LABEL[g], pad=3)
        ax.set_xlabel('Crowd size N'); panel(ax, 'abcd'[k], x=-0.30 if k == 0 else -0.22)
        if k == 0: ax.set_ylabel('RMSE (m)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=7.5)
    fig.subplots_adjust(wspace=0.38); save(fig, 'fig4_adversary_ladder')

# ------------------------------------------------------------------ Fig S1
def figS1():
    d = load('anova_raw_mc50.json'); r = d['results']; Ns = d['config']['Ns']; Trs = d['config']['trolls']
    fig, axes = plt.subplots(1, 2, figsize=(TWO, TWO*0.40))
    for k, tj in enumerate(['circle', 'square']):
        data = {(N, tr): np.array(r[f"{tj}_tr{tr:.2f}_N{N}_T0"]['rmse_raw']) for tr in Trs for N in Ns}
        allv = np.concatenate(list(data.values())); gm = allv.mean(); a, b, n = len(Ns), len(Trs), 50
        ssA = sum(b*n*(np.mean([data[(N, tr)].mean() for tr in Trs])-gm)**2 for N in Ns)
        ssB = sum(a*n*(np.mean([data[(N, tr)].mean() for N in Ns])-gm)**2 for tr in Trs)
        ssT = ((allv-gm)**2).sum(); ssC = sum(n*(v.mean()-gm)**2 for v in data.values()); ssAB = ssC-ssA-ssB; ssE = ssT-ssC
        dfA, dfB = a-1, b-1; dfAB = dfA*dfB; dfE = a*b*(n-1)
        F = [(ssA/dfA)/(ssE/dfE), (ssB/dfB)/(ssE/dfE), (ssAB/dfAB)/(ssE/dfE)]
        p = [stats.f.sf(F[0], dfA, dfE), stats.f.sf(F[1], dfB, dfE), stats.f.sf(F[2], dfAB, dfE)]
        eta = [100*ssA/ssT, 100*ssB/ssT, 100*ssAB/ssT]; eta.append(100-sum(round(e, 1) for e in eta))   # residual as in Table 2
        ax = axes[k]; cols = ['#1f77b4', '#e6821e', '#2ca02c', '#9a9a9a']
        bars = ax.bar(range(4), eta, color=cols, edgecolor='black', linewidth=1.0)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"N\n({eta[0]:.1f}%)", f"tr\n({eta[1]:.1f}%)", f"N×tr\n({eta[2]:.1f}%)", f"Residual\n({eta[3]:.1f}%)"])
        ax.set_ylabel('η² (%)'); ax.set_title(f'{LABEL[tj]} trajectory', pad=3); ax.set_ylim(0, max(eta)*1.22); panel(ax, 'ab'[k])
        for i in range(3):
            if p[i] == 0 or p[i] < 1e-300: lab = r'$P < 10^{-300}$'
            else:
                m, e = f"{p[i]:.2e}".split('e'); lab = rf'$P = {float(m):.1f} \times 10^{{{int(e)}}}$'
            ax.text(i, eta[i]+max(eta)*0.02, lab, ha='center', va='bottom', fontsize=6.5)
    fig.subplots_adjust(wspace=0.35); save(fig, 'figS1_anova_eta2')

# ------------------------------------------------------------------ Fig S2
def figS2():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(TWO, TWO*0.62)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    def box(x, y, w, h, title, body, fc='#f4f4f4'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=fc, ec='black', lw=1.0))
        ax.text(x+w/2, y+h-2.0, title, ha='center', va='top', fontsize=7.5, fontweight='bold')
        ax.text(x+w/2, y+h-7.0, body, ha='center', va='top', fontsize=6.3, linespacing=1.25)
    def arrow(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=9, lw=1.0, color='black'))
    box(26, 80, 48, 18, 'Simulation engine (adversary_ladder.py)',
        '8-direction votes, vector mean · fixed speed 5 m/s · delay 433 ms\nlook-ahead 2 m · 65 s horizon · zigzag: periodic path (corrected)', '#e8eef7')
    box(2, 48, 46, 24, 'Published grid (T0 / T1 / T2)',
        '4 geometries × 10 tr × 12 N × 3 adversary models\nMC = 50: 72,000 runs (zigzag re-simulated: 23,400)\n'
        'raw-run ANOVA 3,600 · speed 3,600\nbehaviour panel 12,000 · lag axis 800 · mechanism 400')
    box(52, 48, 46, 24, 'Revision campaigns',
        'geometry scale sweep 27,000 · radius extension 3,600\npolygon shape sweep 25,200 · large crowds to 3,200: 8,000\n'
        'measured latency τ = 867 ms: 1,600\nvote-noise floor · corner response · localisation (~1,000)')
    box(4, 8, 28, 30, 'Analysis',
        'ceiling fit a + b/√N (OLS, WLS)\nchange-point in b(tr)\ntwo-way ANOVA\nfloor law a(R) = a₀ + c/R\nquadrature e² = e_lag² + e_adv²\nsaturation onset, N_c(δ)', '#eef7e8')
    box(36, 8, 30, 30, 'Main text',
        'Fig 1 paths\nFig 2 b(tr), R² map\nFig 3 RMSE–N to 3,200\nFig 4 adversary ladder\nTables 1–4', '#fbf1e0')
    box(70, 8, 26, 30, 'Supplement and archive',
        'Tables S1–S9\nFigs S1–S5\nZenodo: code, data\n(21 files), figure scripts', '#fbf1e0')
    arrow(38, 80, 25, 72); arrow(62, 80, 75, 72); arrow(18, 48, 18, 38); arrow(75, 48, 26, 38)
    arrow(32, 23, 36, 23); arrow(66, 23, 70, 23)
    save(fig, 'figS2_pipeline')

# ------------------------------------------------------------------ Fig S3
def figS3():
    """Size axis: floor a vs circle radius, 2.5–80 m, with the directly measured vote-noise floor."""
    sw = load('sweep_geometry_scale.json')['results']; ext = load('circle_range_ext.json'); fl = load('vote_noise_floor.json')['results']
    fig, ax = plt.subplots(figsize=(ONE*1.25, ONE*0.95))
    Rx = np.logspace(np.log10(2), np.log10(100), 200)
    for tr, c, mk in [(0.20, '#e6821e', 'o'), (0.40, '#c0508a', 's')]:
        f = ext['fits_full_range'][f"tr{tr:.2f}"]; Rs = np.array(f['R']); a = np.array(f['a'])
        ax.plot(Rs, a, mk, ms=4, color=c, label=f'tr = {int(tr*100)}%')
        q = f['a0+c/R']; ax.plot(Rx, q['a0']+q['c']/Rx, '--', lw=1.0, color=c, label=f"a₀ + c/R, R² = {q['R2']:.3f}")
        fv = fl[f"tr{tr:.2f}"]; ax.axhspan(fv['rmse_mean']-fv['rmse_std'], fv['rmse_mean']+fv['rmse_std'], color=c, alpha=0.12, lw=0)
    ax.set_xscale('log'); ax.set_xlim(1.8, 100); ax.set_xticks([2.5, 5, 10, 20, 40, 80]); ax.set_xticklabels(['2.5', '5', '10', '20', '40', '80'])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel('Circle radius R (m)'); ax.set_ylabel('Error floor a (m)'); ax.legend(loc='upper right', fontsize=6.5)
    ax.text(0.98, 0.30, 'shaded: floor measured directly\nat R = 1,000 m, N = 3,200 (±1 s.d.)', transform=ax.transAxes, ha='right', fontsize=6.5, color='0.4')
    save(fig, 'figS3_size_sweep')

# ------------------------------------------------------------------ Fig S4
def figS4():
    """Shape axis: floor relative to a circle of equal perimeter, and the failed collapse variable."""
    A = load('sweep_polygon_shape_size.json')['analysis']; sides = [3, 4, 6, 8, 12, 24]; vt = 5.0*26/60
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TWO, TWO*0.40), gridspec_kw=dict(wspace=0.32))
    for tr, ls in [(0.20, '-'), (0.40, '--')]:
        for P, c, mk in [(40, '#1f77b4', 'o'), (80, '#e6821e', 's'), (160, '#2ca02c', '^')]:
            ab = A['P2_shape'][f"P{P}_tr{tr:.2f}"]['a_by_n']; rel = 100*(np.array(ab[:6])/ab[6]-1)
            a1.plot(range(6), rel, mk, ls=ls, ms=4, color=c, lw=1.2, label=f'P = {P} m, tr = {int(tr*100)}%')
            lam = [(P/n)/vt for n in sides]; a2.plot(lam, rel, mk, ms=4, color=c, ls='none', mfc=c if tr == 0.2 else 'white')
    a1.axhline(0, color='0.5', lw=1.0); a1.set_xticks(range(6)); a1.set_xticklabels([str(n) for n in sides]); a1.set_xlabel('Number of sides n')
    a1.set_ylabel('Floor relative to circle of equal perimeter (%)'); a1.set_ylim(-20, 100); a1.legend(fontsize=6, ncol=2, loc='upper right', handlelength=2.2); panel(a1, 'a')
    a2.axhline(0, color='0.5', lw=1.0); a2.set_xscale('log'); a2.set_xlabel('Λ = corner spacing / delay travel distance'); a2.set_ylabel('Floor relative to circle (%)')
    r2 = A['P5_collapse']; a2.text(0.03, 0.95, f"log-linear fit: R² = {r2['tr0.20']['R2']:.2f} (tr 20%), {r2['tr0.40']['R2']:.2f} (tr 40%)\nfilled: tr 20%, open: tr 40%",
                                   transform=a2.transAxes, va='top', fontsize=6.5, color='0.35'); a2.set_ylim(-20, 100); a2.set_xticks([1, 3, 10, 30]); a2.set_xticklabels(['1', '3', '10', '30']); a2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); panel(a2, 'b')
    save(fig, 'figS4_shape_sweep')

# ------------------------------------------------------------------ Fig S5
def figS5():
    loc = load('localization_index_ci.json'); order = ['zigzag', 'circle', 'square', 'lemniscate']; conds = [(5, 0.2), (200, 0.2), (200, 0.4)]
    fig, ax = plt.subplots(figsize=(ONE*1.25, ONE*0.9)); w = 0.26; xs = np.arange(4)
    for k, (N, tr) in enumerate(conds):
        v = [loc[f"{g}_N{N}_tr{tr}"]['index'] for g in order]; lo = [loc[f"{g}_N{N}_tr{tr}"]['index']-loc[f"{g}_N{N}_tr{tr}"]['ci'][0] for g in order]
        hi = [loc[f"{g}_N{N}_tr{tr}"]['ci'][1]-loc[f"{g}_N{N}_tr{tr}"]['index'] for g in order]
        ax.bar(xs+(k-1)*w, v, w, yerr=[lo, hi], capsize=2, color=['#c8c8c8', '#8a8a8a', '#3f3f3f'][k], edgecolor='white', linewidth=1.0, label=f'N = {N}, tr = {int(tr*100)}%', error_kw=dict(lw=1.0))
    ax.axhline(1.0, color='0.45', lw=1.0, ls='--'); ax.text(3.35, 1.05, 'uniform\nalong path', fontsize=6.5, color='0.45', ha='right')
    ax.set_xticks(xs); ax.set_xticklabels([LABEL[g] for g in order]); ax.set_ylabel('Error at reversals / error mid-segment'); ax.legend(fontsize=6.5, loc='upper right')
    save(fig, 'figS5_error_localisation')

if __name__ == '__main__':
    want = sys.argv[sys.argv.index('--fig')+1] if '--fig' in sys.argv else 'all'
    todo = {'1': fig1, '2': fig2, '3': fig3, '4': fig4, 'S1': figS1, 'S2': figS2, 'S3': figS3, 'S4': figS4, 'S5': figS5}
    for k, f in todo.items():
        if want in ('all', k): f()
