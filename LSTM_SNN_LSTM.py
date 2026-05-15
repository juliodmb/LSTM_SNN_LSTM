# -*- coding: utf-8 -*-
"""





Created on Mon Apr 13 16:53:40 2026

@author: julio



Demonstração completa e corrigida de todos os componentes do snnTorch 0.9.4
Baseado exclusivamente na documentação oficial:
https://snntorch.readthedocs.io/en/latest/

FLUXO COMPLETO DA REDE:

  Input x[:, step, :]
        ↓
  [***Encoding***] — OPCIONAL, antes do loop temporal
  spikegen.rate / latency / delta
        ↓
  [***Corrente I[t]***] — nn.Linear
  cur = W · X[t]                    ← sem beta, sem alpha no Leaky
        ↓
  [***Potencial de Membrana U[t]***] — snn.Leaky / snn.Synaptic
  Leaky:    U[t+1] = β·U[t] + I_in[t+1]         ← β age AQUI no potencial
  Synaptic: I_syn[t+1] = α·I_syn[t] + I_in[t+1] ← α age AQUI na corrente
            U[t+1]    = β·U[t] + I_syn[t+1]      ← β age AQUI no potencial
        ↓
  [***Threshold U_thr***]
  valor de referência — critério de comparação
        ↓
  [***Heaviside Θ***]
  S[t] = Θ(U[t] - U_thr) → 1 ou 0
  decide os dois lados — não só o 0
        ↓
  [***Reset R[t]***]
  subtract: U[t+1] = β·U[t] + I[t+1] - R·U_thr
  zero:     U[t+1] = β·U[t] + I[t+1] - R·(β·U[t] + I[t+1])
        ↓
  spk → próxima camada (nn.Linear)
  mem → próximo timestep (estado persiste)
        ↓
  ... repete num_steps vezes ...
        ↓
  Loss em mem[-1]
        ↑
  [***Surrogate Gradient***] — só no backward
  substitui dΘ/dU = δ(U - U_thr) por função diferenciável
  padrão da biblioteca: ATan



Pipeline SNN - EBPR PAO/GAO Phase Detection
Dataset: Agtrup BlueKolding
Autor: pipeline base para snnTorch 0.9.4

Estrutura:
  1. Importação e validação do df_rate
  2. Detecção de dead states por inferência de variação
  3. Três encodings: Rate, Delta, Latency
  4. Análise exploratória visual comparativa
  5. Saída pronta para alimentar snnTorch
"""

"""
Pipeline SNN - EBPR PAO/GAO Phase Detection
Dataset: Agtrup BlueKolding
Autor: pipeline base para snnTorch 0.9.4

Estrutura:
  1. Importação e validação do df_rate
  2. Detecção de dead states por inferência de variação
  3. Três encodings: Rate, Delta, Latency
  4. Análise exploratória visual comparativa
  5. Saída pronta para alimentar snnTorch
"""

"""
Pipeline SNN - EBPR PAO/GAO Phase Detection
Dataset: Agtrup BlueKolding — todas as variáveis SCADA
Autor: pipeline base para snnTorch 0.9.4

Colunas do dataset:
  SENSORES (o que o processo mede):
    fosfato_mgL     — PO4 sensor principal — encoding: rate + delta + latency
    oxigenio_mgL    — DO dissolvido       — encoding: rate + delta
    amonia_mgL      — NH4                 — encoding: rate + delta
    sensor_entrada  — caudal entrada      — encoding: rate
    sensor_saida    — caudal saida        — encoding: rate

  ACTUADORES (o que o operador controla):
    metal_natural_Lh    — dosagem metal sem coagulante — encoding: rate
    metal_colocado_Lh   — dosagem metal com coagulante — encoding: rate
    bomba_coagulante_pct — % bomba coagulante          — encoding: rate

  AMBIENTE (condições externas):
    temperatura_C   — temperatura afluente — encoding: rate
    vazao_m3h       — caudal volumétrico   — encoding: rate + delta





Propósito deste script:
  Exploração visual ESTÁTICA dos encodings.
  Não há modelo, não há treino.
  O output é visual — para informar escolhas de hiperparâmetros futuros.

Colunas e encodings aplicados:
  fosfato_mgL         → rate + delta + latency  (biológico principal)
  oxigenio_mgL        → rate + delta + latency  (biológico)
  amonia_mgL          → rate + delta + latency  (biológico)
  metal_natural_Lh    → rate + delta + latency  (actuador contínuo)
  metal_colocado_Lh   → rate + delta + latency  (actuador contínuo)
  sensor_entrada      → rate + delta            (caudal)
  sensor_saida        → rate + delta            (caudal)
  vazao_m3h           → rate + delta            (caudal)
  bomba_coagulante_pct→ rate + delta            (actuador discreto)
  temperatura_C       → rate                    (ambiente lento)

Total canais SNN: 24

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 0. LEITURA DO DATASET REAL
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXCEL_PATH  = os.path.join(_SCRIPT_DIR, 'agtrup_traduzido9.xlsx')

print(f"[LOAD] Lendo: {_EXCEL_PATH}")
df = pd.read_excel(_EXCEL_PATH)
print(f"[LOAD] Shape bruto: {df.shape}")
print(f"[LOAD] Colunas: {df.columns.tolist()}")
print(f"[LOAD] Amostra:\n{df.head(3)}\n")

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICAÇÃO DAS COLUNAS
# ─────────────────────────────────────────────────────────────────────────────
#
# 'biologico'  → rate + delta + latency
#   Variáveis que reflectem actividade bacteriana directa.
#   O TIMING das variações é o sinal diagnóstico PAO/GAO.
#
# 'continuo'   → rate + delta + latency
#   Actuadores que variam continuamente — dosagem de metal.
#   O delta captura quando a dosagem foi alterada.
#   O latency captura há quanto tempo não há dosagem significativa.
#
# 'caudal'     → rate + delta
#   Caudais hidráulicos — relevante a variação mas não o timing de crossing.
#
# 'discreto'   → rate + delta
#   Actuadores com comportamento on/off ou percentual discreto.
#
# 'ambiente'   → rate
#   Temperatura — varia lentamente, delta seria quase sempre zero.

# Todas as colunas recebem os 3 encodings: rate + delta + latency
# A interpretação de cada encoding varia por variável mas a
# representação é gerada para todas — decisão de quais usar
# fica para a fase de hiperparâmetros do modelo SNN.
COL_CONFIG = {
    'fosfato_mgL':          'completo',
    'oxigenio_mgL':         'completo',
    'amonia_mgL':           'completo',
    'metal_natural_Lh':     'completo',
    'metal_colocado_Lh':    'completo',
    'sensor_entrada':       'completo',
    'sensor_saida':         'completo',
    'vazao_m3h':            'completo',
    'bomba_coagulante_pct': 'completo',
    'temperatura_C':        'completo',
}

TODAS_COLUNAS = list(COL_CONFIG.keys())

# ─────────────────────────────────────────────────────────────────────────────
# 1. VALIDAÇÃO E REINDEXAÇÃO TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────

def load_and_validate(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'])
        df = df.set_index('data').sort_index()
    full_index = pd.date_range(start=df.index.min(),
                               end=df.index.max(), freq='2min')
    df = df.reindex(full_index)
    print(f"[VALIDATE] Shape: {df.shape}")
    print(f"[VALIDATE] Período: {df.index.min()} → {df.index.max()}")
    print(f"[VALIDATE] NaN:\n{df[TODAS_COLUNAS].isna().sum()}\n")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 2. DETECÇÃO DE DEAD STATES
# ─────────────────────────────────────────────────────────────────────────────

def detect_dead_states(series: pd.Series,
                       flat_threshold: float = 1e-4,
                       min_steps: int = 5) -> pd.Series:
    dead = series.isna()
    rolling_std = series.ffill().rolling(window=min_steps,
                                         min_periods=min_steps).std()
    return dead | (rolling_std < flat_threshold)

def build_dead_state_mask(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in TODAS_COLUNAS:
        if col in df.columns:
            df[f'{col}_dead'] = detect_dead_states(df[col])
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3. ENCODINGS
# ─────────────────────────────────────────────────────────────────────────────

def rate_encoding(s: pd.Series, dead: pd.Series) -> pd.Series:
    """
    FÓRMULA: rate(t) = clip( (x(t) - Q05) / (Q95 - Q05), 0, 1 )

    NORMALIZAÇÃO Q05/Q95:
      Usa percentis robustos em vez de min/max.
      Min/max são capturados por outliers extremos — comprimem
      99% dos valores para uma faixa pequena.
      Q05/Q95 garante que a distribuição típica ocupa todo [0,1].
      Outliers acima de Q95 ficam acima de 1 e são cortados pelo clip.

    DEAD → 0.0 (silêncio — neurônio não recebe input)
    """
    q05, q95 = s.quantile(0.05), s.quantile(0.95)
    norm = (s - q05) / (q95 - q05 + 1e-8)
    norm = norm.clip(0.0, 1.0)
    norm[dead] = 0.0
    return norm.rename(f'{s.name}_rate')

def delta_encoding(s: pd.Series, dead: pd.Series) -> pd.Series:
    """
    FÓRMULA: delta(t) = clip( (x(t) - x(t-1)) / (MAD * 1.4826), -1, 1 )

    ORIGEM: primeira derivada discreta — captura velocidade de mudança.

    MAD = Median Absolute Deviation = mediana( |x(t) - mediana(x)| )
    Factor 1.4826: converte MAD para equivalente de desvio padrão gaussiano.
    Mais robusto que std porque a mediana não é afectada por outliers.

    DEAD → 0.0. Transição entrada/saída do dead → 0.0 (evita delta artificial).
    """
    sf = s.copy()
    sf[dead] = np.nan
    sf = sf.ffill()
    delta = sf.diff()
    mad = delta.abs().median()
    norm = delta / (mad * 1.4826 + 1e-8)
    norm = norm.clip(-1.0, 1.0)
    norm[dead] = 0.0
    dead_end = dead & ~dead.shift(1).fillna(False)
    norm[dead_end] = 0.0
    return norm.rename(f'{s.name}_delta')

def latency_encoding(s: pd.Series, dead: pd.Series,
                     thr_q: float = 0.65,
                     max_steps: int = 180) -> pd.Series:
    """
    FÓRMULA: lat(t) = 1 - (t - t_ultimo_crossing) / tau_max

    THRESHOLD = quantile(x, 0.65)
      Nível acima do comportamento típico — cruzar este nível
      é um evento biologicamente relevante.

    CROSSING: qualquer passagem pelo threshold (sobe ou desce).

    INVERSÃO INTENCIONAL:
      Crossing agora   → lat = 1.0 → input máximo ao neurônio
      Silêncio longo   → lat → 0.0 → input mínimo
      A SNN aprende que silêncio prolongado é anómalo.

    DEAD → lat = 0.0 (latência máxima — "não sei quando foi o último evento")
    tau_max = 180 steps = 6h = 1 ciclo PAO completo
    """
    threshold = s.quantile(thr_q)
    above = (s > threshold).astype(int)
    crossing = above.diff().fillna(0).abs() > 0
    lat = pd.Series(float(max_steps), index=s.index)
    last = 0
    for i, (_, cross) in enumerate(crossing.items()):
        if cross:
            last = i
        lat.iloc[i] = min(i - last, max_steps)
    lat[dead] = float(max_steps)
    norm = 1.0 - (lat / max_steps)
    return norm.rename(f'{s.name}_latency')

# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCODING DE TODAS AS VARIÁVEIS
# ─────────────────────────────────────────────────────────────────────────────

def encode_all(df: pd.DataFrame):
    """
    Aplica encodings a cada coluna conforme o seu tipo físico.
    Retorna df_enc com todas as representações + lista de canais SNN.
    """
    frames = []
    encoded_cols = []

    for col, tipo in COL_CONFIG.items():
        if col not in df.columns:
            print(f"[ENCODE] AVISO: coluna '{col}' não encontrada — ignorada")
            continue

        s    = df[col]
        dead = df[f'{col}_dead']
        frm  = pd.DataFrame(index=df.index)
        frm[f'{col}_raw']  = s
        frm[f'{col}_dead'] = dead

        # rate — todos os tipos
        frm[f'{col}_rate'] = rate_encoding(s, dead)
        encoded_cols.append(f'{col}_rate')

        # delta — todas as colunas
        frm[f'{col}_delta'] = delta_encoding(s, dead)
        encoded_cols.append(f'{col}_delta')

        # latency — todas as colunas
        frm[f'{col}_latency'] = latency_encoding(s, dead)
        encoded_cols.append(f'{col}_latency')

        frames.append(frm)

    df_enc = pd.concat(frames, axis=1)
    df_enc = df_enc.loc[:, ~df_enc.columns.duplicated()]

    print(f"\n[ENCODE] Canais gerados ({len(encoded_cols)} total):")
    for c in encoded_cols:
        print(f"         {c}")

    return df_enc, encoded_cols

# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZAÇÃO — raw + encodings + histograma lateral
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'raw':     '#e6edf3',
    'rate':    '#58a6ff',
    'delta':   '#3fb950',
    'latency': '#d2a8ff',
    'dead':    '#f85149',
    'grid':    '#21262d',
    'text':    '#8b949e',
}

def _plot_one_column(col: str, tipo: str, df_enc: pd.DataFrame,
                     win: pd.DataFrame, output_dir: str,
                     window_hours: int):
    """
    Gera 1 figura por variável com:
      - Coluna esquerda (2/3): série temporal de cada encoding
      - Coluna direita (1/3): histograma horizontal com skew e kurt

    HISTOGRAMA:
      density=True → normaliza área para 1 (estimativa de densidade)
      orientation='horizontal' → eixo Y = valores, eixo X = densidade
      skew = assimetria da distribuição (pandas .skew())
      kurt = curtose — peso das caudas vs gaussiana (pandas .kurt())
    """
    # Determinar quais encodings esta coluna tem
    encs_disponiveis = []
    for enc in ('rate', 'delta', 'latency'):
        if f'{col}_{enc}' in win.columns:
            encs_disponiveis.append(enc)

    n_rows = 1 + len(encs_disponiveis)  # raw + encodings
    t = np.arange(len(win))

    fig = plt.figure(figsize=(18, 3.5 * n_rows))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(n_rows, 3, figure=fig,
                           hspace=0.5, wspace=0.3,
                           left=0.06, right=0.97,
                           top=0.93, bottom=0.05)

    def _style_ax(ax):
        ax.set_facecolor('#161b22')
        for sp in ax.spines.values():
            sp.set_color(COLORS['grid'])
        ax.tick_params(colors=COLORS['text'], labelsize=7)
        ax.grid(axis='y', color=COLORS['grid'], lw=0.4, alpha=0.6)
        ax.set_xlim(0, len(t))
        xtick_steps = max(int(window_hours / 8 * 30), 1)
        xticks = np.arange(0, len(t), xtick_steps)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{int(x/30)}h' for x in xticks], fontsize=6)

    def _hist_ax(ax, y, color):
        """
        HISTOGRAMA HORIZONTAL:
          Eixo Y = valores do encoding
          Eixo X = densidade (área = 1)
          bins=60 — resolução suficiente para distribuições não-gaussianas

        SKEW: pd.Series.skew() — terceiro momento central normalizado
          > 0: cauda à direita (maioria dos valores baixos, picos raros altos)
          < 0: cauda à esquerda
          = 0: simétrico

        KURT: pd.Series.kurt() — quarto momento central normalizado - 3
          > 0: leptocúrtico — caudas mais pesadas que gaussiana
          < 0: platicúrtico — caudas mais leves
          Fosfato industrial típico: kurt > 5 (muitos outliers extremos)
        """
        ax.set_facecolor('#161b22')
        for sp in ax.spines.values():
            sp.set_color(COLORS['grid'])
        ax.tick_params(colors=COLORS['text'], labelsize=7)
        valid = y[~np.isnan(y)]
        if len(valid) > 0:
            ax.hist(valid, bins=60, color=color, alpha=0.8,
                    orientation='horizontal', density=True)
            skew = pd.Series(valid).skew()
            kurt = pd.Series(valid).kurt()
            ax.text(0.97, 0.97,
                    f'skew: {skew:.2f}\nkurt: {kurt:.2f}',
                    transform=ax.transAxes,
                    fontsize=7, va='top', ha='right',
                    color=COLORS['text'], fontfamily='monospace')
        ax.set_title('distribuição', color=COLORS['text'], fontsize=7, pad=3)
        ax.grid(axis='x', color=COLORS['grid'], lw=0.4, alpha=0.4)

    # ── Linha 0: raw ──────────────────────────────────────────────────────
    ax_ts = fig.add_subplot(gs[0, :2])
    _style_ax(ax_ts)
    raw_vals = win[f'{col}_raw'].values
    ax_ts.plot(t, raw_vals, color=COLORS['raw'], lw=0.8, alpha=0.9)
    dead_vals = win[f'{col}_dead'].values
    dead_idx  = np.where(dead_vals)[0]
    if len(dead_idx) > 0:
        ax_ts.scatter(dead_idx,
                      np.full(len(dead_idx), np.nanmin(raw_vals)),
                      color=COLORS['dead'], s=3, alpha=0.7, zorder=5,
                      label='Dead state')
        ax_ts.legend(fontsize=6, facecolor='#161b22',
                     edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    ax_ts.set_title(f'{col}  [raw]', color='#e6edf3', fontsize=8, loc='left')

    ax_h = fig.add_subplot(gs[0, 2])
    _hist_ax(ax_h, raw_vals, COLORS['raw'])

    # ── Linhas 1+: encodings ──────────────────────────────────────────────
    for i, enc in enumerate(encs_disponiveis):
        row = i + 1
        ax_ts = fig.add_subplot(gs[row, :2])
        _style_ax(ax_ts)
        y = win[f'{col}_{enc}'].values
        color = COLORS[enc]

        if enc == 'delta':
            pos = np.where(y >= 0, y, 0)
            neg = np.where(y < 0, y, 0)
            ax_ts.fill_between(t, pos, alpha=0.75, color='#3fb950',
                               label='Release (+)')
            ax_ts.fill_between(t, neg, alpha=0.75, color='#f85149',
                               label='Uptake (−)')
            ax_ts.axhline(0, color=COLORS['grid'], lw=0.7, ls='--')
            ax_ts.legend(fontsize=6, facecolor='#161b22',
                         edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
        else:
            ax_ts.plot(t, y, color=color, lw=0.9, alpha=0.9)

        ax_ts.set_title(f'{col}  [{enc}]', color='#e6edf3',
                        fontsize=8, loc='left')

        ax_h = fig.add_subplot(gs[row, 2])
        _hist_ax(ax_h, y, color)

    fig.suptitle(
        f'{col}  |  tipo: {tipo}  |  janela: {window_hours}h',
        color='#e6edf3', fontsize=9, y=0.98, fontfamily='monospace'
    )

    out = os.path.join(output_dir, f'enc_{col}.png')
    plt.savefig(out, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] {out}")


def plot_all(df_enc: pd.DataFrame, output_dir: str,
             window_hours: int = 48, start_offset_hours: int = 200):
    start = df_enc.index.min() + pd.Timedelta(hours=start_offset_hours)
    end   = start + pd.Timedelta(hours=window_hours)
    win   = df_enc[start:end]
    print(f"\n[PLOT] Janela: {start} → {end}  ({len(win)} steps)\n")
    for col, tipo in COL_CONFIG.items():
        if f'{col}_raw' in win.columns:
            _plot_one_column(col, tipo, df_enc, win, output_dir, window_hours)

# ─────────────────────────────────────────────────────────────────────────────
# 6. TENSOR FINAL PARA snnTorch
# ─────────────────────────────────────────────────────────────────────────────

def prepare_snn_tensor(df_enc: pd.DataFrame, encoded_cols: list,
                       window_steps: int = 240,
                       stride_steps: int = 30) -> tuple:
    """
    Shape final: (N_janelas, window_steps, n_canais)

    window_steps = 240  →  8h  →  1 ciclo PAO completo a 2min/step
    stride_steps = 30   →  1h  →  sobreposição entre janelas consecutivas

    Por que janelas sobrepostas?
      Tendências PAO→GAO duram dias.
      Com sobreposição de 7h em 8h, a SNN vê a mesma transição
      de múltiplos ângulos — aumenta o número de amostras sem
      perder a continuidade temporal.
    """
    values = df_enc[encoded_cols].fillna(0.0).values.astype(np.float32)
    n_total, n_canais = values.shape
    windows, indices = [], []
    for start in range(0, n_total - window_steps, stride_steps):
        windows.append(values[start:start + window_steps])
        indices.append(start)
    X = np.stack(windows)
    print(f"\n[TENSOR] Shape: {X.shape}")
    print(f"[TENSOR]   N_janelas = {X.shape[0]}")
    print(f"[TENSOR]   timesteps = {X.shape[1]}  ({X.shape[1]*2}min = {X.shape[1]*2/60:.1f}h)")
    print(f"[TENSOR]   n_canais  = {X.shape[2]}")
    print(f"[TENSOR]   Range: [{X.min():.4f}, {X.max():.4f}]")
    return X, np.array(indices)

# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'outputs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Validar e reindexar temporalmente
    df_val = load_and_validate(df)

    # 2. Detectar dead states em todas as colunas
    df_val = build_dead_state_mask(df_val)

    # 3. Calcular todos os encodings
    df_enc, encoded_cols = encode_all(df_val)

    # 4. Visualização — 1 PNG por variável com histograma lateral
    plot_all(df_enc, output_dir=OUTPUT_DIR,
             window_hours=48, start_offset_hours=200)

    # 5. Tensor final para snnTorch
    X, indices = prepare_snn_tensor(df_enc, encoded_cols,
                                    window_steps=240, stride_steps=30)

    print(f"\n[PRONTO] X shape {X.shape} — pronto para torch.FloatTensor(X)")
    print(f"[PRONTO] Próximo passo: snn_model.py")
    print(f"[PRONTO] Ficheiros em: {OUTPUT_DIR}")
    
    
    
# ─────────────────────────────────────────────────────────────────────────────
# 6. MAPA CAUSAL (RATE + LATENCY)
# ─────────────────────────────────────────────────────────────────────────────

def build_causal_map(df_enc: pd.DataFrame,
                     encoded_cols: list,
                     max_lag: int = 60,
                     min_corr: float = 0.30) -> pd.DataFrame:

    """
    Descobre relações causais entre variáveis usando
    cross-correlation temporal.

    max_lag = atraso máximo em steps
      60 steps = 120 min (dataset 2min)

    min_corr = correlação mínima para considerar ligação.
    """

    rate_cols = [c for c in encoded_cols if c.endswith("_rate")]
    lat_cols  = [c for c in encoded_cols if c.endswith("_latency")]

    causal_edges = []

    def scan(cols):

        for a in cols:
            for b in cols:

                if a == b:
                    continue

                best_corr = 0
                best_lag  = 0

                for lag in range(1, max_lag):

                    s1 = df_enc[a]
                    s2 = df_enc[b].shift(-lag)

                    corr = s1.corr(s2)

                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag  = lag

                if abs(best_corr) >= min_corr:

                    causal_edges.append({
                        "cause": a.replace("_rate","").replace("_latency",""),
                        "effect": b.replace("_rate","").replace("_latency",""),
                        "lag_steps": best_lag,
                        "lag_minutes": best_lag * 2,
                        "corr": best_corr,
                        "encoding": a.split("_")[-1]
                    })

    scan(rate_cols)
    scan(lat_cols)

    causal_df = pd.DataFrame(causal_edges)

    if len(causal_df) > 0:
        causal_df = causal_df.sort_values("corr", key=np.abs, ascending=False)

    return causal_df

def plot_causal_graph(causal_df: pd.DataFrame, output_dir: str):

    import networkx as nx

    G = nx.DiGraph()

    for _, r in causal_df.iterrows():

        G.add_edge(
            r["cause"],
            r["effect"],
            weight=r["corr"],
            lag=r["lag_minutes"]
        )

    plt.figure(figsize=(10,8))

    pos = nx.spring_layout(G, k=0.8)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="#58a6ff",
        node_size=2000,
        font_size=8,
        edge_color="#d2a8ff"
    )

    labels = {(u,v):f'{d["lag"]}m' for u,v,d in G.edges(data=True)}

    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=7)

    out = os.path.join(output_dir, "causal_map.png")

    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()

    print(f"[CAUSAL] mapa salvo em {out}")

# 4. Mapa causal do sistema
causal_df = build_causal_map(df_enc, encoded_cols)

print("\n[CAUSAL MAP]")
print(causal_df.head(20))

causal_path = os.path.join(OUTPUT_DIR, "causal_edges.csv")
causal_df.to_csv(causal_path, index=False)


# 3. Calcular todos os encodings
df_enc, encoded_cols = encode_all(df_val)

# ==========================================
# CALCULO DE TRANSFER ENTROPY
# ==========================================

from pyinform.transferentropy import transfer_entropy
from itertools import permutations

cols = [c for c in df_enc.columns if ('rate' in c or 'latency' in c)]

df_te = df_enc[cols].copy()

for c in df_te.columns:
    df_te[c] = pd.qcut(df_te[c], q=10, labels=False, duplicates="drop")

df_te = df_te.fillna(0).astype(int)

def compute_te_lags(x, y, max_lag=20):

    tes = []

    for lag in range(1, max_lag):

        try:
            te = transfer_entropy(x[:-lag], y[lag:], k=1)
            tes.append(te)
        except:
            tes.append(0)

    return max(tes)

results = []

for source, target in permutations(df_te.columns, 2):

    x = df_te[source].values
    y = df_te[target].values

    te = compute_te_lags(x, y)

    if te > 0.02:
        results.append({
            "source": source,
            "target": target,
            "transfer_entropy": te
        })

te_df = pd.DataFrame(results).sort_values(
    "transfer_entropy",
    ascending=False
)

print("\nTransfer Entropy detectada:\n")
print(te_df)

# ==========================================

# 4. Visualização
plot_all(df_enc, output_dir=OUTPUT_DIR,
         window_hours=48, start_offset_hours=200)

# 5. Tensor final
X, indices = prepare_snn_tensor(df_enc, encoded_cols,
                                window_steps=240, stride_steps=30)