# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 18:34:32 2025

@author: julio
"""

"""
Created on Mon Apr 13 16:53:40 2026

@author: julio
"""

"""
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
# CLASSIFICAÇÃO DAS COLUNAS POR PAPEL FÍSICO
# ─────────────────────────────────────────────────────────────────────────────
#
# Isso determina QUAL encoding faz sentido para cada variável:
#
#  SENSORES BIOLÓGICOS — capturam a dinâmica das bactérias
#    → rate + delta + latency (3 representações)
#    → delta captura variação brusca de actividade
#    → latency captura timing dos eventos biológicos
#
#  SENSORES DE CAUDAL — capturam a carga hidráulica
#    → rate + delta (2 representações)
#    → delta captura variações bruscas de carga
#    → latency não faz sentido — não há "crossing" biológico
#
#  ACTUADORES — o que o operador fez (input de controlo)
#    → apenas rate (1 representação)
#    → são decisões discretas — delta seria ruído
#    → incluídos como contexto operacional, não como sinal biológico
#
#  AMBIENTE — condições externas lentas
#    → apenas rate (1 representação)
#    → variam lentamente — delta seria quase sempre zero

COLUNAS_BIOLOGICAS   = ['fosfato_mgL', 'oxigenio_mgL', 'amonia_mgL']
COLUNAS_CAUDAL       = ['sensor_entrada', 'sensor_saida', 'vazao_m3h']
COLUNAS_ACTUADORES   = ['metal_natural_Lh', 'metal_colocado_Lh', 'bomba_coagulante_pct']
COLUNAS_AMBIENTE     = ['temperatura_C']

TODAS_COLUNAS = COLUNAS_BIOLOGICAS + COLUNAS_CAUDAL + COLUNAS_ACTUADORES + COLUNAS_AMBIENTE

# ─────────────────────────────────────────────────────────────────────────────
# 1. VALIDAÇÃO E REINDEXAÇÃO TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────

def load_and_validate(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Converte coluna 'data' para datetime index e reindexca para 2min exactos.

    Por que reindexar?
      O SCADA pode ter gaps — timesteps em falta por falha de comunicação.
      Sem reindexação, a SNN veria o tempo "saltar" sem saber.
      Com reindexação, os gaps ficam como NaN explícito — informação, não lacuna.
    """
    df = df_raw.copy()

    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'])
        df = df.set_index('data').sort_index()

    # Reindexar para frequência perfeita de 2 minutos
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='2min'
    )
    df = df.reindex(full_index)

    print(f"[VALIDATE] Shape após reindexação: {df.shape}")
    print(f"[VALIDATE] Período: {df.index.min()} → {df.index.max()}")
    nan_summary = df[TODAS_COLUNAS].isna().sum()
    print(f"[VALIDATE] NaN por coluna:\n{nan_summary}\n")

    return df

# ─────────────────────────────────────────────────────────────────────────────
# 2. DETECÇÃO DE DEAD STATES
# ─────────────────────────────────────────────────────────────────────────────

def detect_dead_states(
    series: pd.Series,
    flat_threshold: float = 1e-4,
    min_duration_steps: int = 5
) -> pd.Series:
    """
    Detecta dead states em duas camadas:

    Camada 1 — NaN explícito:
      x(t) = NaN → sensor offline → dead = True
      Origem: gaps do SCADA ou reindexação

    Camada 2 — sensor travado:
      std( x[t-4 : t] ) < flat_threshold → variação zero → dead = True
      Origem: sensor a transmitir valor constante por falha de hardware
      
      Por que rolling std de janela 5?
        5 steps = 10 minutos — janela mínima para distinguir
        um período genuinamente estável de um sensor travado.
        flat_threshold=1e-4 é conservador — abaixo da resolução
        de qualquer sensor analógico de qualidade industrial.
    """
    dead = series.isna()

    rolling_std = series.ffill().rolling(
        window=min_duration_steps,
        min_periods=min_duration_steps
    ).std()
    dead = dead | (rolling_std < flat_threshold)

    return dead


def build_dead_state_mask(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[f'{col}_dead'] = detect_dead_states(df[col])
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3. ENCODINGS — FÓRMULAS E ORIGEM
# ─────────────────────────────────────────────────────────────────────────────

def rate_encoding(series: pd.Series, dead: pd.Series) -> pd.Series:
    """
    FÓRMULA:
      rate(t) = clip( (x(t) - Q05) / (Q95 - Q05), 0, 1 )

    ORIGEM:
      Neurociência computacional — o neurônio LIF dispara com frequência
      proporcional à intensidade do input. Portanto o input deve estar
      em [0,1] onde 1 = disparo máximo, 0 = silêncio.

    NORMALIZAÇÃO:
      Usa percentis Q05 e Q95 em vez de min/max.
      Por quê?
        min e max capturam outliers extremos.
        Se o fosfato atinge 6 mg/L uma vez em 2 anos,
        e normalmente fica abaixo de 1 mg/L, uma normalização
        min/max comprime 99% dos valores para [0, 0.17].
        Com Q05/Q95, a distribuição típica ocupa todo [0,1]
        e os outliers ficam acima de 1 — depois cortados pelo clip.

    DEAD STATES → 0.0 (silêncio — neurônio não recebe input)
    """
    s = series.copy()
    q05, q95 = s.quantile(0.05), s.quantile(0.95)
    norm = (s - q05) / (q95 - q05 + 1e-8)
    norm = norm.clip(0.0, 1.0)
    norm[dead] = 0.0
    return norm.rename(f'{series.name}_rate')


def delta_encoding(series: pd.Series, dead: pd.Series) -> pd.Series:
    """
    FÓRMULA:
      δ(t) = clip( (x(t) - x(t-1)) / (MAD × 1.4826), -1, 1 )

    ORIGEM:
      Processamento de sinais — primeira derivada discreta.
      Captura a VELOCIDADE de mudança, não o valor absoluto.
      Crucial para EBPR: o que distingue PAO de GAO não é
      o nível de fosfato mas a rapidez e simetria do ciclo.

    NORMALIZAÇÃO:
      MAD = Median Absolute Deviation = mediana(|δ(t) - mediana(δ)|)
      Factor 1.4826: constante matemática que converte MAD para
      equivalente de desvio padrão gaussiano (σ ≈ 1.4826 × MAD).
      
      Por que MAD e não std?
        std é sensível a outliers — um spike extremo infla o std
        e comprime toda a distribuição para perto de zero.
        MAD usa a mediana — robusta a outliers.
      
      Por que clip em ±1 e não ±3σ?
        Variações acima de 1 MAD normalizado já são eventos relevantes.
        Preservar a magnitude exacta acima desse threshold não
        acrescenta informação para a SNN — o que importa é
        QUE aconteceu um evento brusco, não exactamente o quão brusco.

    DEAD STATES → 0.0 (sem variação detectável)
    TRANSIÇÃO PARA DEAD → 0.0 (evita delta artificial na entrada/saída)
    """
    s = series.copy()
    s[dead] = np.nan
    s = s.ffill()
    delta = s.diff()
    mad = delta.abs().median()
    norm = delta / (mad * 1.4826 + 1e-8)
    norm = norm.clip(-1.0, 1.0)
    norm[dead] = 0.0
    # Zerar delta artificial no primeiro step após dead state
    dead_end = dead & ~dead.shift(1).fillna(False)
    norm[dead_end] = 0.0
    return norm.rename(f'{series.name}_delta')


def latency_encoding(
    series: pd.Series,
    dead: pd.Series,
    threshold_quantile: float = 0.65,
    max_latency_steps: int = 180
) -> pd.Series:
    """
    FÓRMULA:
      lat(t) = 1 - ( t - t_ultimo_crossing ) / τ_max

    ORIGEM:
      SNNs biológicas — rank-order coding e latency coding.
      O cérebro codifica a urgência de um estímulo no tempo
      até o primeiro spike, não na sua amplitude.

    O QUE É O THRESHOLD?
      threshold = quantile(x, 0.65)
      O nível que o fosfato cruza quando há actividade PAO relevante.
      Percentil 65: acima do comportamento típico mas abaixo dos picos extremos.

    O QUE É UM CROSSING?
      Qualquer momento em que o sinal passa de abaixo para acima
      do threshold (ou vice-versa). Cada crossing é um "evento".

    POR QUE FICA "AO CONTRÁRIO"?
      A inversão é intencional:
        Crossing agora    → lat = 1.0 → neurônio recebe input máximo
        Nenhum crossing   → lat → 0.0 → neurônio recebe input mínimo
      Isso codifica URGÊNCIA — quando há actividade recente, o
      input é alto. Quando o processo está em silêncio prolongado,
      o input cai. A SNN aprende que silêncio prolongado é anómalo.

    DEAD STATES → max_latency (lat_norm = 0.0)
      "Não sei quando foi o último evento" = latência máxima = urgência zero.
      τ_max = 180 steps = 6h = duração típica de 1 ciclo PAO completo.
    """
    s = series.copy()
    threshold = s.quantile(threshold_quantile)
    above = (s > threshold).astype(int)
    crossing = above.diff().fillna(0).abs() > 0

    latency = pd.Series(float(max_latency_steps), index=s.index)
    last = 0
    for i, (_, is_cross) in enumerate(crossing.items()):
        if is_cross:
            last = i
        latency.iloc[i] = min(i - last, max_latency_steps)

    latency[dead] = float(max_latency_steps)
    lat_norm = 1.0 - (latency / max_latency_steps)
    return lat_norm.rename(f'{series.name}_latency')


def encode_column(df: pd.DataFrame, col: str, tipo: str) -> pd.DataFrame:
    """
    Aplica os encodings adequados a uma coluna conforme o seu tipo físico.

    tipo='biologico'  → rate + delta + latency (3 canais)
    tipo='caudal'     → rate + delta           (2 canais)
    tipo='actuador'   → rate                   (1 canal)
    tipo='ambiente'   → rate                   (1 canal)
    """
    s = df[col]
    dead = df[f'{col}_dead']
    result = pd.DataFrame(index=df.index)
    result[f'{col}_raw']  = s
    result[f'{col}_dead'] = dead
    result[f'{col}_rate'] = rate_encoding(s, dead)
    if tipo in ('biologico', 'caudal'):
        result[f'{col}_delta'] = delta_encoding(s, dead)
    if tipo == 'biologico':
        result[f'{col}_latency'] = latency_encoding(s, dead)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCODING DE TODAS AS VARIÁVEIS
# ─────────────────────────────────────────────────────────────────────────────

def encode_all_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica encode_column a todas as variáveis do dataset.
    
    Resultado final — canais que a SNN vai receber:
      fosfato_mgL     : rate + delta + latency  (3 canais — variável principal)
      oxigenio_mgL    : rate + delta            (2 canais)
      amonia_mgL      : rate + delta            (2 canais)
      sensor_entrada  : rate + delta            (2 canais)
      sensor_saida    : rate + delta            (2 canais)
      vazao_m3h       : rate + delta            (2 canais)
      metal_natural_Lh    : rate               (1 canal)
      metal_colocado_Lh   : rate               (1 canal)
      bomba_coagulante_pct: rate               (1 canal)
      temperatura_C       : rate               (1 canal)
      ─────────────────────────────────────────────────
      TOTAL: 17 canais de input para a SNN
    """
    frames = []

    for col in COLUNAS_BIOLOGICAS:
        frames.append(encode_column(df, col, 'biologico'))

    for col in COLUNAS_CAUDAL:
        frames.append(encode_column(df, col, 'caudal'))

    for col in COLUNAS_ACTUADORES:
        frames.append(encode_column(df, col, 'actuador'))

    for col in COLUNAS_AMBIENTE:
        frames.append(encode_column(df, col, 'ambiente'))

    df_enc = pd.concat(frames, axis=1)

    # Remover colunas duplicadas (raw e dead já calculados)
    df_enc = df_enc.loc[:, ~df_enc.columns.duplicated()]

    print(f"[ENCODE] Canais de encoding gerados:")
    encoded_cols = [c for c in df_enc.columns
                    if c.endswith(('_rate', '_delta', '_latency'))]
    for c in encoded_cols:
        print(f"         {c}")
    print(f"[ENCODE] Total canais SNN: {len(encoded_cols)}")

    return df_enc, encoded_cols

# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZAÇÃO COMPARATIVA — TODAS AS VARIÁVEIS
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_sensors(
    df_enc: pd.DataFrame,
    window_hours: int = 48,
    start_offset_hours: int = 200,
    output_dir: str = '.',
):
    """
    Gera um gráfico por variável mostrando raw + encodings aplicados.
    Cada ficheiro PNG tem o nome da variável.
    """
    start = df_enc.index.min() + pd.Timedelta(hours=start_offset_hours)
    end   = start + pd.Timedelta(hours=window_hours)
    win   = df_enc[start:end]

    config = {
        'fosfato_mgL':          ('biologico', ['rate', 'delta', 'latency']),
        'oxigenio_mgL':         ('caudal',    ['rate', 'delta']),
        'amonia_mgL':           ('biologico', ['rate', 'delta', 'latency']),
        'sensor_entrada':       ('caudal',    ['rate', 'delta']),
        'sensor_saida':         ('caudal',    ['rate', 'delta']),
        'vazao_m3h':            ('caudal',    ['rate', 'delta']),
        'metal_natural_Lh':     ('actuador',  ['rate']),
        'metal_colocado_Lh':    ('actuador',  ['rate']),
        'bomba_coagulante_pct': ('actuador',  ['rate']),
        'temperatura_C':        ('ambiente',  ['rate']),
    }

    colors = {
        'raw':     '#e6edf3',
        'rate':    '#58a6ff',
        'delta':   '#3fb950',
        'latency': '#d2a8ff',
        'dead':    '#f85149',
        'grid':    '#21262d',
        'text':    '#8b949e',
    }

    for col, (tipo, encs) in config.items():
        n_rows = 1 + len(encs)  # raw + encodings
        fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3 * n_rows))
        fig.patch.set_facecolor('#0d1117')
        if n_rows == 1:
            axes = [axes]

        t = np.arange(len(win))

        # Raw
        ax = axes[0]
        ax.set_facecolor('#161b22')
        for sp in ax.spines.values(): sp.set_color(colors['grid'])
        ax.tick_params(colors=colors['text'], labelsize=7)
        raw_vals = win[f'{col}_raw'].values
        ax.plot(t, raw_vals, color=colors['raw'], lw=0.8)
        dead_vals = win[f'{col}_dead'].values
        dead_idx  = np.where(dead_vals)[0]
        if len(dead_idx) > 0:
            ax.scatter(dead_idx, np.full(len(dead_idx), np.nanmin(raw_vals)),
                       color=colors['dead'], s=3, alpha=0.7, zorder=5)
        ax.set_title(f'{col}  [raw]', color='#e6edf3', fontsize=8, loc='left')
        ax.grid(axis='y', color=colors['grid'], lw=0.4)

        # Encodings
        for i, enc in enumerate(encs):
            ax = axes[i + 1]
            ax.set_facecolor('#161b22')
            for sp in ax.spines.values(): sp.set_color(colors['grid'])
            ax.tick_params(colors=colors['text'], labelsize=7)
            y = win[f'{col}_{enc}'].values

            if enc == 'delta':
                pos = np.where(y >= 0, y, 0)
                neg = np.where(y < 0, y, 0)
                ax.fill_between(t, pos, alpha=0.75, color='#3fb950')
                ax.fill_between(t, neg, alpha=0.75, color='#f85149')
                ax.axhline(0, color=colors['grid'], lw=0.6, ls='--')
            else:
                ax.plot(t, y, color=colors[enc], lw=0.8)

            # Estatísticas
            valid = y[~np.isnan(y)]
            skew = pd.Series(valid).skew()
            kurt = pd.Series(valid).kurt()
            ax.text(0.99, 0.97, f'skew:{skew:.2f} kurt:{kurt:.2f}',
                    transform=ax.transAxes, fontsize=6, va='top', ha='right',
                    color=colors['text'], fontfamily='monospace')
            ax.set_title(f'{col}  [{enc}]', color='#e6edf3', fontsize=8, loc='left')
            ax.grid(axis='y', color=colors['grid'], lw=0.4)
            ax.set_xlim(0, len(t))

            # Xticks como horas
            xtick_steps = int(window_hours / 8 * 30)
            xticks = np.arange(0, len(t), max(xtick_steps, 1))
            ax.set_xticks(xticks)
            ax.set_xticklabels([f'{int(x/30)}h' for x in xticks], fontsize=6)

        fig.suptitle(f'{col}  |  janela {window_hours}h  |  tipo: {tipo}',
                     color='#e6edf3', fontsize=9, y=1.01, fontfamily='monospace')
        plt.tight_layout()

        out = os.path.join(output_dir, f'enc_{col}.png')
        plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print(f"[PLOT] Guardado: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. TENSOR FINAL PARA snnTorch
# ─────────────────────────────────────────────────────────────────────────────

def prepare_snn_tensor(
    df_enc: pd.DataFrame,
    encoded_cols: list,
    window_steps: int = 240,
    stride_steps: int = 30,
) -> np.ndarray:
    """
    Constrói o tensor de entrada para a SNN.

    Shape final: (N_janelas, window_steps, n_canais)
      N_janelas   = número de janelas deslizantes
      window_steps = 240 steps = 8h = 1 ciclo PAO completo
      n_canais    = 17 (todos os encodings de todas as variáveis)

    Por que janelas deslizantes?
      A SNN processa uma janela temporal de cada vez.
      Com stride=30 (1h), janelas consecutivas partilham 7h de contexto.
      Isso é crucial para capturar tendências multi-ciclo (PAO→GAO leva dias).

    Por que window_steps=240?
      1 ciclo PAO = ~8h a 2min/step = 240 steps.
      A SNN tem de ver pelo menos 1 ciclo completo para
      aprender a relação release-anaeróbio/uptake-aeróbio.
    """
    values = df_enc[encoded_cols].fillna(0.0).values.astype(np.float32)
    n_total, n_canais = values.shape

    windows, indices = [], []
    for start in range(0, n_total - window_steps, stride_steps):
        windows.append(values[start:start + window_steps])
        indices.append(start)

    X = np.stack(windows)  # (N, 240, 17)

    print(f"\n[TENSOR] Shape final: {X.shape}")
    print(f"[TENSOR]   N_janelas  = {X.shape[0]}")
    print(f"[TENSOR]   timesteps  = {X.shape[1]}  ({X.shape[1]*2} min = {X.shape[1]*2/60:.1f}h)")
    print(f"[TENSOR]   n_canais   = {X.shape[2]}")
    print(f"[TENSOR]   Range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"[TENSOR]   Canais: {encoded_cols}")

    return X, np.array(indices)

# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'outputs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Redireciona savefig para OUTPUT_DIR local
    import matplotlib
    _orig_savefig = matplotlib.pyplot.savefig
    def _savefig_local(fname, *args, **kwargs):
        if str(fname).startswith('/mnt/'):
            fname = os.path.join(OUTPUT_DIR, os.path.basename(str(fname)))
        return _orig_savefig(fname, *args, **kwargs)
    matplotlib.pyplot.savefig = _savefig_local

    # 1. Validar e reindexar
    df_val = load_and_validate(df)

    # 2. Detectar dead states em todas as colunas
    df_val = build_dead_state_mask(df_val, TODAS_COLUNAS)

    # 3. Calcular todos os encodings
    df_enc, encoded_cols = encode_all_columns(df_val)

    # 4. Visualização — 1 gráfico por variável
    plot_all_sensors(df_enc, window_hours=48, start_offset_hours=200,
                     output_dir=OUTPUT_DIR)

    # 5. Tensor final para snnTorch
    X, indices = prepare_snn_tensor(df_enc, encoded_cols,
                                    window_steps=240, stride_steps=30)

    print(f"\n[PRONTO] X shape {X.shape} — pronto para torch.FloatTensor(X)")
    print(f"[PRONTO] Próximo passo: snn_model.py — DataLoader + LIF + treino")
    print(f"[PRONTO] Ficheiros em: {OUTPUT_DIR}")
