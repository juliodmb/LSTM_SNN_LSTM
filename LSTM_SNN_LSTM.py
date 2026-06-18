# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PIPELINE SNN — EBPR PAO/GAO Phase Detection                                ║
║  Dataset: Agtrup BlueKolding SCADA                                          ║
║  Base: snnTorch 0.9.4                                                       ║
║                                                                              ║
║  FLUXO COMPLETO DA REDE (para referência futura):                           ║
║                                                                              ║
║  Input x[:, step, :]          shape: (batch, timestep, n_canais)            ║
║        ↓                                                                     ║
║  [Encoding] — feito AQUI neste script, antes do modelo                      ║
║  rate / delta / latency → valores em [0,1] ou [-1,1]                        ║
║        ↓                                                                     ║
║  [nn.Linear] → Corrente I[t]                                                ║
║  cur = W · X[t]    ← multiplicação matricial simples                        ║
║        ↓                                                                     ║
║  [snn.Leaky] → Potencial de Membrana U[t]                                   ║
║  U[t+1] = β·U[t] + I[t+1]                                                  ║
║  β = e^(-Δt/τ_m)  ← decay físico, NÃO é parâmetro livre                   ║
║        ↓                                                                     ║
║  [Heaviside Θ] → Spike S[t]                                                 ║
║  S[t] = 1 se U[t] ≥ U_thr, senão 0                                         ║
║  Decide AMBOS os lados — spike E silêncio                                   ║
║        ↓                                                                     ║
║  [Reset subtract]                                                            ║
║  U[t+1] = β·U[t] + I[t+1] - S[t]·U_thr                                    ║
║  Preserva excesso acima do threshold — fisicamente correcto para PAO        ║
║        ↓                                                                     ║
║  spk → próxima camada | mem → próximo timestep                              ║
║        ↓  (repete num_steps=240 vezes — 1 ciclo PAO completo)               ║
║  Loss em spk_out (ce_rate_loss ou ce_temporal_loss)                         ║
║        ↑                                                                     ║
║  [Surrogate Gradient ATan] — só no backward pass                            ║
║  substitui dΘ/dU = δ(U-U_thr) por arco-tangente diferenciável              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PROPÓSITO DESTE SCRIPT:
  Fase 1 — Pré-processamento e exploração visual ESTÁTICA.
  Não há modelo, não há treino aqui.
  Output: tensor X shape (N, 240, 30) pronto para snn_model.py

MÉTRICAS IMPLEMENTADAS:
  Spearman ρ    — preservação de ordem após normalização
  Cobertura     — % de valores no range esperado [0,1] ou [-1,1]
  Entropia H    — riqueza informacional da distribuição normalizada
  Transfer Entropy — causalidade temporal entre variáveis (bits)
  Cross-correlation — lag de causalidade em minutos
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# rate/delta/latency são implementados do zero — sem import externo de SNN aqui
# snnTorch entra no próximo script (snn_model.py)
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np          # álgebra — arrays, diff, clip, histogram
import pandas as pd         # séries temporais, rolling, quantile, corr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
warnings.filterwarnings('ignore')

import torch  # só para detectar GPU agora — será usado intensamente no modelo

# ─────────────────────────────────────────────────────────────────────────────
# DETECÇÃO DE GPU
# torch.cuda.is_available() consulta o driver NVIDIA via CUDA runtime
# Se False → cai para CPU, treino mais lento mas funcional
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {DEVICE}")

if torch.cuda.is_available():
    print(f"[GPU]  {torch.cuda.get_device_name(0)}")
    print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    # benchmark=True: cuDNN testa kernels e escolhe o mais rápido para o shape actual
    # Custo: ~30s na primeira iteração. Ganho: 10-30% velocidade depois
    torch.backends.cudnn.benchmark = True
    # high precision: usa TF32 em vez de FP32 em multiplicações matriciais
    # Precision ligeiramente menor, velocidade ~2x em Ampere+
    torch.set_float32_matmul_precision("high")

# ─────────────────────────────────────────────────────────────────────────────
# 0. LEITURA DO DATASET
# ─────────────────────────────────────────────────────────────────────────────
# os.path.abspath(__file__) → caminho absoluto deste script
# os.path.dirname(...)      → directoria do script
# os.path.join(..., 'agtrup_traduzido9.xlsx') → caminho do Excel na mesma pasta
# Isso garante que o script funciona independentemente de onde é executado
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXCEL_PATH  = os.path.join(_SCRIPT_DIR, 'agtrup_traduzido9.xlsx')

print(f"[LOAD] Lendo: {_EXCEL_PATH}")
df = pd.read_excel(_EXCEL_PATH)
print(f"[LOAD] Shape bruto: {df.shape}")         # (525600, 11)
print(f"[LOAD] Colunas: {df.columns.tolist()}")
print(f"[LOAD] Amostra:\n{df.head(3)}\n")

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICAÇÃO DAS COLUNAS
# 'completo' → os 3 encodings: rate + delta + latency
# Decisão: gerar tudo agora, escolher na fase de hiperparâmetros do modelo
# ─────────────────────────────────────────────────────────────────────────────
COL_CONFIG = {
    # Variáveis biológicas — reflectem actividade PAO/GAO directamente
    # Timing das variações é a assinatura diagnóstica
    'fosfato_mgL':          'completo',   # PO4 — variável principal
    'oxigenio_mgL':         'completo',   # DO  — aeração define fases
    'amonia_mgL':           'completo',   # NH4 — ciclo N correlacionado

    # Actuadores contínuos — variam em tempo real
    # delta captura quando a dosagem mudou
    # latency captura há quanto tempo sem dosagem significativa
    'metal_natural_Lh':     'completo',
    'metal_colocado_Lh':    'completo',

    # Caudais — carga hidráulica afecta HRT → afecta PAO/GAO
    'sensor_entrada':       'completo',
    'sensor_saida':         'completo',
    'vazao_m3h':            'completo',

    # Actuador on/off — gerado completo para comparação visual
    # latency pode mostrar estrutura se há dosagem periódica
    'bomba_coagulante_pct': 'completo',

    # Ambiente — varia lentamente (horas/dias)
    # delta ≈ 0 quase sempre, latency ≈ 0 quase sempre
    # mantido completo para o gráfico mostrar empiricamente essa limitação
    'temperatura_C':        'completo',
}

TODAS_COLUNAS = list(COL_CONFIG.keys())  # lista ordenada das 10 colunas

# ─────────────────────────────────────────────────────────────────────────────
# 1. VALIDAÇÃO E REINDEXAÇÃO TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────
def load_and_validate(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    OBJECTIVO: garantir que o índice temporal é perfeito antes de qualquer cálculo.

    pd.to_datetime()    → converte string 'YYYY-MM-DD HH:MM:SS' para Timestamp
    set_index('data')   → define a coluna data como índice do DataFrame
    sort_index()        → ordena cronologicamente (SCADA pode ter registos fora de ordem)

    pd.date_range(freq='2min') → cria sequência perfeita de timestamps a cada 2 minutos
    df.reindex(full_index)     → alinha o df com essa sequência perfeita
      Se havia um timestamp em falta no SCADA → aparece como NaN
      Se havia timestamps duplicados → o reindex mantém o primeiro

    POR QUE ISTO É CRÍTICO PARA A SNN:
      O neurônio LIF integra input ao longo de timesteps consecutivos.
      Se há um gap silencioso (timestamps em falta) e não o marcamos,
      a SNN vê dois timesteps adjacentes que na realidade têm horas de distância.
      O potencial de membrana U[t] acumulado num período seria incorrecto.
      Com NaN explícito → dead state → silêncio → U decai correctamente.
    """
    df = df_raw.copy()
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'])
        df = df.set_index('data').sort_index()

    # Reindexar para frequência exacta de 2 minutos
    full_index = pd.date_range(start=df.index.min(),
                               end=df.index.max(), freq='2min')
    df = df.reindex(full_index)

    print(f"[VALIDATE] Shape: {df.shape}")
    print(f"[VALIDATE] Período: {df.index.min()} → {df.index.max()}")
    print(f"[VALIDATE] NaN por coluna:\n{df[TODAS_COLUNAS].isna().sum()}\n")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 2. DETECÇÃO DE DEAD STATES
# ─────────────────────────────────────────────────────────────────────────────
def detect_dead_states(series: pd.Series,
                       flat_threshold: float = 1e-4,
                       min_steps: int = 5) -> pd.Series:
    """
    DUAS CAMADAS DE DETECÇÃO:

    CAMADA 1 — NaN explícito:
      series.isna() → True onde o valor é NaN
      Origem: gaps de comunicação SCADA ou reindexação

    CAMADA 2 — Sensor travado:
      series.ffill() → propaga o último valor válido nos NaN
        (necessário para o rolling não propagar NaN)
      .rolling(window=5).std() → desvio padrão numa janela de 5 steps = 10min
        std ≈ 0 → sensor transmitindo valor constante → travado

      flat_threshold=1e-4:
        Abaixo da resolução analógica de qualquer sensor industrial de qualidade.
        Um sensor real sempre tem pelo menos 0.001 de ruído entre leituras.
        Se std < 0.0001 por 10 minutos → com certeza está travado.

    OPERADOR |:
      Combina as duas camadas — dead se NaN OU se std < threshold.

    PARA A SNN:
      dead=True → encoding = 0.0 (silêncio total)
      dead=True em latency → latência máxima ("não vi evento há muito tempo")
      O modelo aprende que silêncio prolongado tem significado próprio.
    """
    dead = series.isna()  # Camada 1: NaN explícito

    # Camada 2: sensor travado — rolling std
    rolling_std = series.ffill().rolling(window=min_steps,
                                          min_periods=min_steps).std()
    return dead | (rolling_std < flat_threshold)


def build_dead_state_mask(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica detect_dead_states a cada coluna e adiciona coluna '{col}_dead'.
    Resultado: df com 10 colunas originais + 10 colunas booleanas de dead state.
    """
    df = df.copy()
    for col in TODAS_COLUNAS:
        if col in df.columns:
            df[f'{col}_dead'] = detect_dead_states(df[col])
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3. ENCODINGS — FÓRMULAS, ORIGEM E INTERPRETAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def rate_encoding(s: pd.Series, dead: pd.Series) -> pd.Series:
    """
    FÓRMULA:
      rate(t) = clip( (x(t) - Q05) / (Q95 - Q05 + ε), 0, 1 )

    MATEMÁTICA:
      Q05 = percentil 5  → valor abaixo do qual estão 5% dos dados
      Q95 = percentil 95 → valor abaixo do qual estão 95% dos dados
      (x - Q05) / (Q95 - Q05) → normalização linear para [0,1] tipicamente
      clip(0,1) → garante que outliers não saem do range

    POR QUE Q05/Q95 E NÃO MIN/MAX:
      min/max são capturados por outliers extremos.
      Se fosfato atinge 6 mg/L uma vez em 2 anos mas normalmente fica em 0.5,
      min/max comprime 99% dos dados para [0, 0.083].
      Com Q05/Q95, a distribuição típica ocupa todo [0,1].

    PARA A SNN (snnTorch):
      rate(t) ∈ [0,1] → usado directamente como intensidade de input.
      Neurônio LIF recebe cur = W * rate(t).
      Se rate=1.0 → cur máximo → U sobe rapidamente → spikes frequentes.
      Se rate=0.0 → cur=0 → U decai com β → silêncio.
      Equivalente biológico: frequência de disparo ∝ intensidade do estímulo.

    ε = 1e-8: evita divisão por zero quando Q05 ≈ Q95 (variável constante).
    DEAD → 0.0: silêncio absoluto, sem input ao neurônio.
    """
    q05, q95 = s.quantile(0.05), s.quantile(0.95)
    norm = (s - q05) / (q95 - q05 + 1e-8)
    norm = norm.clip(0.0, 1.0)
    norm[dead] = 0.0
    return norm.rename(f'{s.name}_rate')


def delta_encoding(s: pd.Series, dead: pd.Series) -> pd.Series:
    """
    FÓRMULA:
      δ(t) = clip( (x(t) - x(t-1)) / (MAD × 1.4826 + ε), -1, 1 )

    MATEMÁTICA:
      x(t) - x(t-1)  → diferença discreta de primeira ordem (derivada discreta)
                        captura a VELOCIDADE de mudança, não o valor

      MAD = median( |δ(t) - median(δ)| )
        Mediana dos desvios absolutos em torno da mediana.
        Não é afectada por outliers — um spike extremo não infla a MAD.

      1.4826: constante que satisfaz E[MAD] = σ para distribuição gaussiana.
        Torna a normalização comparável entre variáveis com escalas diferentes.
        Deriva do facto de que para N(0,σ): MAD = σ × Φ^-1(3/4) ≈ 0.6745σ
        → 1/0.6745 ≈ 1.4826

    POR QUE DELTA É O ENCODING MAIS IMPORTANTE PARA PAO/GAO:
      O que distingue PAO de GAO não é o nível de fosfato absoluto
      mas a simetria e velocidade do ciclo release/uptake.
      PAO saudável: δ grande positivo (release rápido) seguido de δ grande negativo (uptake rápido)
      GAO crescente: δ positivo atenuado, δ negativo ausente ou tardio

    DEAD STATES — tratamento cuidadoso:
      sf[dead] = NaN → impede que o valor do dead state contamine o delta
      sf.ffill() → propaga o último valor válido (evita delta artificial ao entrar no dead)
      dead_end → primeiro step após o dead state: delta zerado
        Sem este passo, ao sair do dead state haveria um delta enorme
        (diferença entre o valor antes do dead e depois) que seria artificial.

    clip(-1,1): variações acima de 1 MAD normalizado são eventos relevantes.
      Preservar a magnitude exacta não acrescenta informação para a SNN.
    """
    sf = s.copy()
    sf[dead] = np.nan         # isola dead states
    sf = sf.ffill()           # propaga último valor válido
    delta = sf.diff()         # derivada discreta: x(t) - x(t-1)
    mad = delta.abs().median() # MAD robusto
    norm = delta / (mad * 1.4826 + 1e-8)  # normalização MAD
    norm = norm.clip(-1.0, 1.0)
    norm[dead] = 0.0          # silêncio nos dead states
    dead_end = dead & ~dead.shift(1).fillna(False)  # primeiro step pós-dead
    norm[dead_end] = 0.0      # zera delta artificial na saída do dead state
    return norm.rename(f'{s.name}_delta')


def latency_encoding(s: pd.Series, dead: pd.Series,
                     thr_q: float = 0.65,
                     max_steps: int = 180) -> pd.Series:
    """
    FÓRMULA:
      lat_norm(t) = 1 - (t - t_ultimo_crossing) / τ_max

    MATEMÁTICA:
      threshold = quantile(x, 0.65)
        Nível no percentil 65 — acima do comportamento típico.
        Para o fosfato: o nível que separa "normal baixo" de "evento de release/uptake"

      above(t) = 1 se x(t) > threshold, senão 0
        Sinal binário: está acima ou abaixo do nível crítico

      crossing(t) = |above(t) - above(t-1)| > 0
        Detecta qualquer passagem pelo threshold (sobe OU desce)
        .diff() → diferença entre timesteps consecutivos
        .abs() > 0 → qualquer mudança de estado = crossing

      t - t_ultimo_crossing → timesteps desde o último evento
        Contador que reseta a 0 a cada crossing e cresce linearmente

      τ_max = 180 steps = 6h = duração típica de 1 ciclo PAO completo
        Define o horizonte temporal de memória do encoding

      1 - (...)/τ_max → inversão:
        crossing agora (contador=0) → lat_norm = 1.0 = input máximo
        sem crossing há τ_max steps → lat_norm = 0.0 = silêncio

    POR QUE FICA "AO CONTRÁRIO" DO RATE:
      Dente de serra: cai a pique (crossing) e sobe lentamente (tempo a acumular).
      Invertido: desce lentamente (tempo passando) e sobe a pique (crossing).
      Isso codifica URGÊNCIA TEMPORAL — evento recente = alta actividade.
      A SNN aprende: "latency baixo por muito tempo" = processo em silêncio anómalo.

    DIAGNÓSTICO PAO→GAO:
      PAO saudável: dentes de serra regulares com período ~6h
      GAO crescente: dentes de serra ficam mais espaçados (crossings mais raros)
        porque o release/uptake enfraquece antes do PO4 efluente subir.
      Este alargamento progressivo é detectável pela SNN dias antes da falha química.

    DEAD STATES → lat_norm = 0.0 (latência máxima):
      "Não sei quando foi o último evento" = urgência zero.
    """
    threshold = s.quantile(thr_q)
    above = (s > threshold).astype(int)          # sinal binário acima/abaixo
    crossing = above.diff().fillna(0).abs() > 0  # detecta mudanças de estado
    lat = pd.Series(float(max_steps), index=s.index)  # inicializa com máximo
    last = 0
    for i, (_, cross) in enumerate(crossing.items()):
        if cross:
            last = i               # reseta o contador a cada crossing
        lat.iloc[i] = min(i - last, max_steps)  # steps desde o último crossing
    lat[dead] = float(max_steps)   # dead state = latência máxima
    norm = 1.0 - (lat / max_steps) # inversão: crossing recente = 1.0
    return norm.rename(f'{s.name}_latency')

# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCODING DE TODAS AS VARIÁVEIS → 30 CANAIS
# ─────────────────────────────────────────────────────────────────────────────
def encode_all(df: pd.DataFrame):
    """
    Aplica rate + delta + latency a todas as 10 colunas.
    Output: df_enc com 30 colunas de encoding + lista dos nomes.

    30 canais = 10 variáveis × 3 encodings
    Cada encoding captura uma dimensão diferente da mesma variável:
      rate    → valor actual normalizado (O QUÊ)
      delta   → velocidade de mudança   (COM QUE RAPIDEZ)
      latency → tempo desde último evento (QUANDO foi o último evento)
    """
    frames = []
    encoded_cols = []

    for col, tipo in COL_CONFIG.items():
        if col not in df.columns:
            print(f"[ENCODE] AVISO: '{col}' não encontrada — ignorada")
            continue

        s    = df[col]
        dead = df[f'{col}_dead']
        frm  = pd.DataFrame(index=df.index)
        frm[f'{col}_raw']  = s       # sinal bruto (para visualização)
        frm[f'{col}_dead'] = dead    # máscara de dead state

        # Rate — todos os tipos
        frm[f'{col}_rate'] = rate_encoding(s, dead)
        encoded_cols.append(f'{col}_rate')

        # Delta — todas as colunas
        frm[f'{col}_delta'] = delta_encoding(s, dead)
        encoded_cols.append(f'{col}_delta')

        # Latency — todas as colunas
        frm[f'{col}_latency'] = latency_encoding(s, dead)
        encoded_cols.append(f'{col}_latency')

        frames.append(frm)

    df_enc = pd.concat(frames, axis=1)
    df_enc = df_enc.loc[:, ~df_enc.columns.duplicated()]

    print(f"\n[ENCODE] {len(encoded_cols)} canais gerados:")
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


def _hist_ax(ax, y, color):
    """
    HISTOGRAMA HORIZONTAL com métricas de distribuição.

    density=True:
      Normaliza o histograma para que a área total = 1.
      Eixo X = densidade de probabilidade (não contagem).
      Permite comparar distribuições de variáveis com escalas diferentes.

    orientation='horizontal':
      Eixo Y = valores do encoding (mesma escala que o gráfico temporal ao lado)
      Eixo X = densidade → lê-se "quantos valores estão neste nível"

    SKEWNESS (pd.Series.skew()):
      Terceiro momento central normalizado: E[(X-μ)³] / σ³
      > 0: cauda à direita — maioria dos valores baixos, picos raros altos
            (típico do fosfato industrial: passa 90% do tempo baixo)
      < 0: cauda à esquerda
      = 0: distribuição simétrica
      |skew| > 1: assimetria severa → considerar transformação adicional

    KURTOSIS (pd.Series.kurt()):
      Quarto momento central normalizado - 3 (excess kurtosis)
      E[(X-μ)⁴] / σ⁴ - 3
      > 0: leptocúrtico — caudas mais pesadas que gaussiana
            (eventos extremos mais frequentes do que o esperado)
            Fosfato industrial típico: kurt > 5
      < 0: platicúrtico — caudas mais leves
      = 0: mesokúrtico (gaussiano)

    INTERPRETAÇÃO PARA SNN:
      Alta kurtosis → muitos outliers → o clip do delta é justificado
      Alta skewness → distribuição assimétrica → Q05/Q95 é melhor que min/max
      Distribuição uniforme (kurt≈-1.2) → encoding bem calibrado → alta entropia
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


def _plot_one_column(col, tipo, df_enc, win, output_dir, window_hours):
    encs = [e for e in ('rate', 'delta', 'latency')
            if f'{col}_{e}' in win.columns]
    n_rows = 1 + len(encs)
    t = np.arange(len(win))

    fig = plt.figure(figsize=(18, 3.5 * n_rows))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(n_rows, 3, figure=fig,
                           hspace=0.5, wspace=0.3,
                           left=0.06, right=0.97, top=0.93, bottom=0.05)

    def _style(ax):
        ax.set_facecolor('#161b22')
        for sp in ax.spines.values():
            sp.set_color(COLORS['grid'])
        ax.tick_params(colors=COLORS['text'], labelsize=7)
        ax.grid(axis='y', color=COLORS['grid'], lw=0.4, alpha=0.6)
        ax.set_xlim(0, len(t))
        xt = max(int(window_hours / 8 * 30), 1)
        ticks = np.arange(0, len(t), xt)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{int(x/30)}h' for x in ticks], fontsize=6)

    # Raw
    ax0 = fig.add_subplot(gs[0, :2])
    _style(ax0)
    rv = win[f'{col}_raw'].values
    ax0.plot(t, rv, color=COLORS['raw'], lw=0.8, alpha=0.9)
    di = np.where(win[f'{col}_dead'].values)[0]
    if len(di):
        ax0.scatter(di, np.full(len(di), np.nanmin(rv)),
                    color=COLORS['dead'], s=3, alpha=0.7,
                    zorder=5, label='Dead state')
        ax0.legend(fontsize=6, facecolor='#161b22',
                   edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    ax0.set_title(f'{col}  [raw]', color='#e6edf3', fontsize=8, loc='left')
    _hist_ax(fig.add_subplot(gs[0, 2]), rv, COLORS['raw'])

    # Encodings
    for i, enc in enumerate(encs):
        ax = fig.add_subplot(gs[i+1, :2])
        _style(ax)
        y = win[f'{col}_{enc}'].values
        if enc == 'delta':
            ax.fill_between(t, np.where(y>=0,y,0), alpha=0.75,
                            color='#3fb950', label='Release (+)')
            ax.fill_between(t, np.where(y<0,y,0), alpha=0.75,
                            color='#f85149', label='Uptake (−)')
            ax.axhline(0, color=COLORS['grid'], lw=0.7, ls='--')
            ax.legend(fontsize=6, facecolor='#161b22',
                      edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
        else:
            ax.plot(t, y, color=COLORS[enc], lw=0.9, alpha=0.9)
        ax.set_title(f'{col}  [{enc}]', color='#e6edf3', fontsize=8, loc='left')
        _hist_ax(fig.add_subplot(gs[i+1, 2]), y, COLORS[enc])

    fig.suptitle(f'{col}  |  tipo: {tipo}  |  janela: {window_hours}h',
                 color='#e6edf3', fontsize=9, y=0.98, fontfamily='monospace')
    out = os.path.join(output_dir, f'enc_{col}.png')
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] {out}")


def plot_all(df_enc, output_dir, window_hours=48, start_offset_hours=200):
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
def prepare_snn_tensor(df_enc, encoded_cols,
                       window_steps=240, stride_steps=30):
    """
    JANELAS DESLIZANTES:
      window_steps = 240 steps = 8h = 1 ciclo PAO completo a 2min/step
        A SNN precisa de ver pelo menos 1 ciclo para aprender
        a relação release-anaeróbio / uptake-aeróbio.

      stride_steps = 30 steps = 1h
        Deslocamento entre janelas consecutivas.
        Sobreposição = 240 - 30 = 210 steps = 7h
        Janelas consecutivas partilham 7h de contexto.
        Isto aumenta N (mais amostras) sem perder continuidade temporal.

    SHAPE FINAL: (N, 240, 30)
      N   = número de janelas ≈ (525600 - 240) / 30 ≈ 17512
      240 = timesteps por janela (num_steps da SNN)
      30  = canais de input (10 variáveis × 3 encodings)

    PRÓXIMO PASSO (snn_model.py):
      X_tensor = torch.FloatTensor(X).to(DEVICE)
      # shape: (N, 240, 30)
      # A SNN vai iterar nos 240 timesteps e actualizar U[t] a cada passo
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
# 7. MÉTRICAS DE QUALIDADE DAS NORMALIZAÇÕES
# ─────────────────────────────────────────────────────────────────────────────
def metricas_normalizacao(df_val, df_enc, output_dir):
    """
    TRÊS MÉTRICAS — cada uma captura uma dimensão diferente da qualidade:

    ══ SPEARMAN ρ ══
      Fórmula: ρ = 1 - (6 Σd²) / (n(n²-1))
        d = diferença de rank entre x(t) original e x(t) encoding
      Mede se a ORDEM RELATIVA dos valores foi preservada.
      ρ = 1.0: ordem perfeitamente preservada (transformação monotónica)
      ρ < 0.95: a normalização alterou quem é maior/menor que quem
      Rate deve ser sempre ρ = 1.0 (é monotónica por construção — preserva rank)
      Delta e latency podem ter ρ < 1 — são transformações não-monotónicas
      INTERPRETAÇÃO: ρ baixo no delta não é necessariamente mau —
        o delta captura uma dimensão diferente (velocidade, não valor)

    ══ COBERTURA ══
      % de valores dentro do range esperado: [0,1] para rate/latency, [-1,1] para delta
      100%: todos os valores no range — encoding bem calibrado
      < 95%: muitos outliers extremos — clip está a descartar informação relevante
      ACÇÃO: se cobertura baixa no delta, considera aumentar o clip para [-2,2]

    ══ ENTROPIA DE SHANNON ══
      H = -Σ p(x) × log₂(p(x))    (bits)
      Mede a riqueza informacional da distribuição após normalização.
      H_max = log₂(n_bins) = log₂(50) ≈ 5.64 bits (distribuição uniforme perfeita)
      H_norm = H / H_max ∈ [0,1]

      H_norm ≈ 1.0: distribuição uniforme → máxima informação para a SNN
        Cada valor do encoding é aproximadamente tão provável como qualquer outro
      H_norm ≈ 0.0: distribuição muito concentrada → canal pouco informativo
        A SNN recebe quase sempre o mesmo valor — não há o que aprender

      INTERPRETAÇÃO PRÁTICA:
        Rate com H_norm baixo → variável passa quase sempre pelo mesmo valor
          (ex: metal_natural_Lh que está a zero maior parte do tempo)
        Latency com H_norm alto → cruzamentos frequentes e distribuídos
          (ex: fosfato com ciclos regulares PAO)
        Delta com H_norm ≈ 0.5 → variações simétricas e diversas → ideal

    LINHA LARANJA NO GRÁFICO:
      Marca 0.95 = limiar de qualidade mínima aceitável para Spearman e Cobertura.
    """
    from scipy.stats import spearmanr
    from scipy.stats import entropy as scipy_entropy

    print("\n" + "="*75)
    print("MÉTRICAS DE QUALIDADE DAS NORMALIZAÇÕES")
    print("="*75)
    print(f"{'Coluna':<32} {'Enc':<9} {'Spearman':>9} {'Cobertura':>10} {'Entropia':>9}")
    print("-"*75)

    resultados = []
    for col in TODAS_COLUNAS:
        if col not in df_val.columns:
            continue
        raw = df_val[col].dropna()
        for enc in ('rate', 'delta', 'latency'):
            enc_col = f'{col}_{enc}'
            if enc_col not in df_enc.columns:
                continue
            enc_vals = df_enc[enc_col].dropna()
            idx = raw.index.intersection(enc_vals.index)
            if len(idx) < 100:
                continue
            rho, _ = spearmanr(raw[idx].values, enc_vals[idx].values)
            lo, hi = (-1.0, 1.0) if enc == 'delta' else (0.0, 1.0)
            cobertura = ((enc_vals[idx] >= lo) & (enc_vals[idx] <= hi)).mean()
            counts, _ = np.histogram(enc_vals[idx].values, bins=50)
            counts = counts[counts > 0]
            H = scipy_entropy(counts / counts.sum(), base=2)
            H_norm = H / np.log2(50)
            print(f"{col:<32} {enc:<9} {rho:>9.4f} {cobertura:>9.1%} {H_norm:>9.4f}")
            resultados.append({'coluna': col, 'encoding': enc,
                               'spearman': rho, 'cobertura': cobertura,
                               'entropia_norm': H_norm})

    print("="*75)
    df_res = pd.DataFrame(resultados)
    if df_res.empty:
        return df_res

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0d1117')
    enc_colors = {'rate': '#58a6ff', 'delta': '#3fb950', 'latency': '#d2a8ff'}
    colunas = df_res['coluna'].unique()
    x = np.arange(len(colunas))
    w = 0.25

    metr = [
        ('spearman',      'Spearman ρ\n(preservação de ordem)',       (0.0, 1.05)),
        ('cobertura',     'Cobertura no range\n[0,1] ou [-1,1]',      (0.0, 1.05)),
        ('entropia_norm', 'Entropia normalizada H\n(riqueza de info)', (0.0, 1.05)),
    ]
    for ax, (metric, title, ylim) in zip(axes, metr):
        ax.set_facecolor('#161b22')
        for sp in ax.spines.values(): sp.set_color('#21262d')
        ax.tick_params(colors='#8b949e', labelsize=7)
        for i, enc in enumerate(('rate', 'delta', 'latency')):
            sub = df_res[df_res['encoding'] == enc]
            vals = [sub[sub['coluna']==c][metric].values[0]
                    if len(sub[sub['coluna']==c]) > 0 else 0.0
                    for c in colunas]
            ax.bar(x + i*w, vals, w, color=enc_colors[enc], alpha=0.85, label=enc)
        ax.set_xticks(x + w)
        ax.set_xticklabels(colunas, rotation=45, ha='right',
                           fontsize=6, color='#8b949e')
        ax.set_ylim(*ylim)
        ax.set_title(title, color='#e6edf3', fontsize=8, pad=6)
        ax.legend(fontsize=7, facecolor='#161b22',
                  edgecolor='#21262d', labelcolor='#8b949e')
        ax.grid(axis='y', color='#21262d', lw=0.5)
        # Linha de referência 0.95 — limiar mínimo aceitável
        ax.axhline(0.95, color='#ffa657', lw=0.8, ls='--', alpha=0.7,
                   label='limiar 0.95')

    fig.suptitle('Qualidade das Normalizações — Spearman | Cobertura | Entropia',
                 color='#e6edf3', fontsize=10, y=1.01, fontfamily='monospace')
    plt.tight_layout()
    out = os.path.join(output_dir, 'metricas_normalizacao.png')
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] {out}")
    return df_res

# ─────────────────────────────────────────────────────────────────────────────
# 8. ANÁLISE DE FALHAS DE SENSOR
# ─────────────────────────────────────────────────────────────────────────────
def analise_falhas_sensor(df_val, df_enc, output_dir):
    """
    QUATRO PERSPECTIVAS:

    1. FREQUÊNCIA E DURAÇÃO por variável
       Identifica sensores não-confiáveis que precisam de tratamento especial.
       Sensor com >5% do tempo em dead state → considerar imputação ou exclusão.

    2. CORRELAÇÃO ENTRE DEAD STATES (heatmap)
       Correlação de Pearson entre as séries booleanas de dead state.
       Alta correlação entre múltiplos sensores → falha de INFRA (rede, servidor)
       Correlação próxima de zero → falha de HARDWARE individual do sensor

    3. SIMULTANEIDADE
       Nº de sensores em dead state ao mesmo tempo.
       ≥ 3 simultâneos → quase certamente falha de comunicação, não hardware.

    4. DISTRIBUIÇÃO TEMPORAL (hora e mês)
       Dead states concentrados em horas → manutenção programada
       Dead states concentrados em meses → condições sazonais adversas
       Dead states uniformes → falhas aleatórias de hardware
    """
    dead_cols = [c for c in df_enc.columns if c.endswith('_dead')]
    resumo = []
    for dc in dead_cols:
        col = dc.replace('_dead', '')
        dead_s = df_enc[dc].astype(bool)
        n_dead = dead_s.sum()
        pct = n_dead / len(dead_s) * 100
        runs, in_dead, si = [], False, 0
        for i, v in enumerate(dead_s.values):
            if v and not in_dead:
                in_dead, si = True, i
            elif not v and in_dead:
                in_dead = False
                if (i - si) >= 3:
                    runs.append(i - si)
        n_runs  = len(runs)
        med_dur = np.median(runs) * 2 if runs else 0
        max_dur = np.max(runs)    * 2 if runs else 0
        print(f"{col:<30} | eventos: {n_runs:>5} | "
              f"mediana: {med_dur:>6.0f}min | máximo: {max_dur:>7.0f}min | "
              f"dead: {pct:>5.2f}%")
        resumo.append({'coluna': col, 'n_eventos': n_runs,
                       'mediana_min': med_dur, 'max_min': max_dur,
                       'pct_dead': pct})
    df_r = pd.DataFrame(resumo)

    # Gráfico 1: resumo por variável
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0d1117')
    x = np.arange(len(df_r))
    for ax in axes:
        ax.set_facecolor('#161b22')
        for sp in ax.spines.values(): sp.set_color('#21262d')
        ax.tick_params(colors='#8b949e', labelsize=7)
        ax.grid(axis='y', color='#21262d', lw=0.5)

    axes[0].bar(x, df_r['n_eventos'], color='#f85149', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_r['coluna'], rotation=45, ha='right',
                             fontsize=6, color='#8b949e')
    axes[0].set_title('Nº de eventos de dead state', color='#e6edf3', fontsize=8)

    axes[1].bar(x, df_r['mediana_min'], color='#ffa657', alpha=0.85, label='mediana')
    axes[1].bar(x, df_r['max_min'], color='#f85149', alpha=0.3, label='máximo')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_r['coluna'], rotation=45, ha='right',
                             fontsize=6, color='#8b949e')
    axes[1].set_title('Duração (min)', color='#e6edf3', fontsize=8)
    axes[1].legend(fontsize=7, facecolor='#161b22',
                   edgecolor='#21262d', labelcolor='#8b949e')

    axes[2].bar(x, df_r['pct_dead'], color='#d2a8ff', alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df_r['coluna'], rotation=45, ha='right',
                             fontsize=6, color='#8b949e')
    axes[2].set_title('% tempo em dead state', color='#e6edf3', fontsize=8)
    # Linha de referência: acima de 5% o sensor é não-confiável
    axes[2].axhline(5, color='#ffa657', lw=0.8, ls='--', alpha=0.7,
                    label='limite 5%')
    axes[2].legend(fontsize=7, facecolor='#161b22',
                   edgecolor='#21262d', labelcolor='#8b949e')

    fig.suptitle('Falhas de Sensor — Frequência | Duração | Cobertura',
                 color='#e6edf3', fontsize=10, y=1.01, fontfamily='monospace')
    plt.tight_layout()
    out1 = os.path.join(output_dir, 'falhas_sensor_resumo.png')
    plt.savefig(out1, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] {out1}")

    # Gráfico 2: correlação e simultaneidade
    dead_matrix = pd.DataFrame({
        dc.replace('_dead', ''): df_enc[dc].astype(int)
        for dc in dead_cols if dc in df_enc.columns
    })
    dead_matrix['n_simult'] = dead_matrix.sum(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(18, 8))
    fig.patch.set_facecolor('#0d1117')

    corr = dead_matrix.drop(columns='n_simult').corr()
    ax = axes[0]
    ax.set_facecolor('#161b22')
    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right',
                       fontsize=7, color='#8b949e')
    ax.set_yticklabels(corr.columns, fontsize=7, color='#8b949e')
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.values[i,j]:.2f}',
                    ha='center', va='center', fontsize=6, color='#0d1117')
    plt.colorbar(im, ax=ax, fraction=0.02)
    ax.set_title('Correlação entre dead states — verde alto = falha de infra',
                 color='#e6edf3', fontsize=8, pad=6)

    ax2 = axes[1]
    ax2.set_facecolor('#161b22')
    for sp in ax2.spines.values(): sp.set_color('#21262d')
    ax2.tick_params(colors='#8b949e', labelsize=7)
    t = np.arange(len(dead_matrix))
    ax2.fill_between(t, dead_matrix['n_simult'].values,
                     alpha=0.7, color='#f85149')
    ax2.axhline(3, color='#ffa657', lw=0.8, ls='--', alpha=0.7,
                label='≥3 simultâneos = falha de infra provável')
    ax2.set_title('Sensores em dead state simultaneamente',
                  color='#e6edf3', fontsize=8, pad=6)
    ax2.set_ylabel('N sensores', color='#8b949e', fontsize=7)
    ax2.grid(axis='y', color='#21262d', lw=0.5)
    ax2.legend(fontsize=7, facecolor='#161b22',
               edgecolor='#21262d', labelcolor='#8b949e')

    fig.suptitle('Coincidência de Falhas — Correlação | Simultaneidade',
                 color='#e6edf3', fontsize=10, y=1.01, fontfamily='monospace')
    plt.tight_layout()
    out2 = os.path.join(output_dir, 'falhas_sensor_coincidencia.png')
    plt.savefig(out2, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] {out2}")

    # Gráfico 3: distribuição por hora e mês
    dead_matrix.index = df_enc.index
    dead_matrix['hora'] = dead_matrix.index.hour
    dead_matrix['mes']  = dead_matrix.index.month
    sensor_cols = [c for c in dead_matrix.columns
                   if c not in ('n_simult', 'hora', 'mes')]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('#0d1117')
    for ax, agr, tit, cor in zip(
        axes,
        ['hora', 'mes'],
        ['Dead states por hora do dia', 'Dead states por mês'],
        ['#58a6ff', '#3fb950']
    ):
        ax.set_facecolor('#161b22')
        for sp in ax.spines.values(): sp.set_color('#21262d')
        ax.tick_params(colors='#8b949e', labelsize=7)
        grp = dead_matrix.groupby(agr)[sensor_cols].sum().sum(axis=1)
        ax.bar(grp.index, grp.values, color=cor, alpha=0.85)
        ax.set_title(tit, color='#e6edf3', fontsize=8, pad=6)
        ax.set_xlabel(agr, color='#8b949e', fontsize=7)
        ax.grid(axis='y', color='#21262d', lw=0.5)

    fig.suptitle('Distribuição Temporal das Falhas — Hora | Mês',
                 color='#e6edf3', fontsize=10, y=1.01, fontfamily='monospace')
    plt.tight_layout()
    out3 = os.path.join(output_dir, 'falhas_sensor_temporal.png')
    plt.savefig(out3, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] {out3}")
    return df_r

# ─────────────────────────────────────────────────────────────────────────────
# 9. ANÁLISE SAZONAL — DISTRIBUIÇÃO POR MÊS
# ─────────────────────────────────────────────────────────────────────────────
def analise_sazonal(df_val, output_dir):
    """
    BOXPLOT POR MÊS para cada variável.

    OBJECTIVO PAO/GAO:
      PAO favorecido no Inverno (Dez-Fev): temperatura baixa, HRT estável
      GAO favorecido no Verão (Jun-Ago): temperatura alta, carbono em excesso
      Se fosfato efluente é sistematicamente mais alto no verão →
        confirma hipótese sazonal sem necessidade de teste de DNA.

    ELEMENTOS DO BOXPLOT:
      Linha laranja = mediana mensal → tendência central
      Caixa = IQR (25º a 75º percentil) → variabilidade típica
      Whiskers = 1.5 × IQR → range aceitável
      Pontos vermelhos = outliers (fora de 1.5 × IQR) → eventos extremos

    LINHA VERDE TRACEJADA:
      Mediana mensal conectada → tendência sazonal ao longo do ano.
      Se for uma curva com pico no verão → sazonalidade PAO/GAO confirmada.

    PARA O MODELO:
      Se houver sazonalidade clara, considerar adicionar 'mes' e 'hora_do_dia'
      como features adicionais no tensor de input da SNN.
      Alternativa: treinar modelos separados por estação.
    """
    df_plot = df_val[TODAS_COLUNAS].copy()
    df_plot['mes'] = df_val.index.month
    meses = ['Jan','Fev','Mar','Abr','Mai','Jun',
             'Jul','Ago','Set','Out','Nov','Dez']

    fig, axes = plt.subplots(len(TODAS_COLUNAS), 1,
                             figsize=(16, 4 * len(TODAS_COLUNAS)))
    fig.patch.set_facecolor('#0d1117')

    for ax, col in zip(axes, TODAS_COLUNAS):
        ax.set_facecolor('#161b22')
        for sp in ax.spines.values(): sp.set_color('#21262d')
        ax.tick_params(colors='#8b949e', labelsize=7)
        dados = [df_plot[df_plot['mes']==m][col].dropna().values
                 for m in range(1, 13)]
        ax.boxplot(
            dados, positions=range(1, 13), widths=0.6, patch_artist=True,
            boxprops=dict(facecolor='#58a6ff', alpha=0.5, linewidth=0.8),
            medianprops=dict(color='#ffa657', linewidth=1.5),
            whiskerprops=dict(color='#8b949e', linewidth=0.8),
            capprops=dict(color='#8b949e', linewidth=0.8),
            flierprops=dict(marker='.', color='#f85149',
                            markersize=2, alpha=0.4)
        )
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(meses, fontsize=7, color='#8b949e')
        ax.set_title(f'{col}  — distribuição por mês',
                     color='#e6edf3', fontsize=8, loc='left')
        ax.grid(axis='y', color='#21262d', lw=0.5)
        medianas = [np.median(d) if len(d)>0 else np.nan for d in dados]
        ax.plot(range(1, 13), medianas, color='#3fb950', lw=1.2, ls='--',
                alpha=0.8, marker='o', markersize=3, label='mediana mensal')
        ax.legend(fontsize=6, facecolor='#161b22',
                  edgecolor='#21262d', labelcolor='#8b949e')

    fig.suptitle('Sazonalidade — Distribuição por Mês',
                 color='#e6edf3', fontsize=11, y=1.005, fontfamily='monospace')
    plt.tight_layout()
    out = os.path.join(output_dir, 'sazonalidade_mensal.png')
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. MAPA CAUSAL POR CROSS-CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
def build_causal_map(df_enc, encoded_cols, max_lag=60, min_corr=0.30):
    """
    CROSS-CORRELATION TEMPORAL:
      Para cada par (A, B) de variáveis, testa se A no instante t
      prediz B no instante t+lag.

      corr(A[t], B[t+lag]) = correlação de Pearson entre A e B deslocado lag steps

      Se |corr| ≥ min_corr no lag L → A causa B com atraso L × 2 minutos
      Escolhe o lag com maior |corr| como o atraso dominante.

    LIMITAÇÃO:
      Cross-correlation detecta associação linear com atraso.
      Não é causalidade de Granger formal — não controla para confundidores.
      Para relações não-lineares, usar Transfer Entropy (secção 11).

    max_lag = 60 steps = 120 minutos:
      Atraso máximo testado. Se a causa leva mais de 2h a produzir efeito,
      não será detectada. Para o processo EBPR com ciclos de 6-8h,
      considerar aumentar para max_lag=180 (6h).

    min_corr = 0.30:
      Correlação mínima para reportar uma ligação.
      r=0.30 → R²=9% → explica 9% da variância.
      Para processos biológicos ruidosos, este limiar é razoável.
    """
    rate_cols = [c for c in encoded_cols if c.endswith("_rate")]
    lat_cols  = [c for c in encoded_cols if c.endswith("_latency")]
    causal_edges = []

    def scan(cols):
        for a in cols:
            for b in cols:
                if a == b:
                    continue
                best_corr, best_lag = 0, 0
                for lag in range(1, max_lag):
                    # shift(-lag): desloca B para a esquerda → B está "lag steps no futuro"
                    corr = df_enc[a].corr(df_enc[b].shift(-lag))
                    if abs(corr) > abs(best_corr):
                        best_corr, best_lag = corr, lag
                if abs(best_corr) >= min_corr:
                    causal_edges.append({
                        "cause":       a.replace("_rate","").replace("_latency",""),
                        "effect":      b.replace("_rate","").replace("_latency",""),
                        "lag_steps":   best_lag,
                        "lag_minutes": best_lag * 2,
                        "corr":        best_corr,
                        "encoding":    a.split("_")[-1]
                    })
    scan(rate_cols)
    scan(lat_cols)

    causal_df = pd.DataFrame(causal_edges)
    if len(causal_df) > 0:
        causal_df = causal_df.sort_values("corr", key=np.abs, ascending=False)
    return causal_df


def plot_causal_graph(causal_df, output_dir):
    """
    GRAFO DIRIGIDO:
      Nó = variável
      Aresta A→B = A causa B com o lag em minutos indicado na aresta
      Espessura/cor da aresta proporcional à correlação

    INTERPRETAÇÃO PARA PAO/GAO:
      Se metal_natural → fosfato com lag=30min:
        A dosagem de metal afecta o fosfato 30 minutos depois
      Se fosfato → oxigenio com lag=15min:
        O consumo de oxigénio pelos PAOs responde 15 min após o release de fosfato
      Lags longos (>60min) entre variáveis biológicas → processo lento → GAO suspeito
    """
    try:
        import networkx as nx
    except ImportError:
        print("[CAUSAL] networkx não instalado — instala com: pip install networkx")
        return

    G = nx.DiGraph()
    for _, r in causal_df.iterrows():
        G.add_edge(r["cause"], r["effect"],
                   weight=r["corr"], lag=r["lag_minutes"])

    plt.figure(figsize=(12, 9))
    plt.gca().set_facecolor('#161b22')
    plt.gcf().patch.set_facecolor('#0d1117')
    pos = nx.spring_layout(G, k=1.2, seed=42)
    nx.draw(G, pos, with_labels=True,
            node_color='#58a6ff', node_size=2500,
            font_size=7, font_color='#0d1117',
            edge_color='#d2a8ff', arrows=True,
            arrowsize=20, width=1.5)
    edge_labels = {(u,v): f'{d["lag"]}m' for u,v,d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=6, font_color='#ffa657')
    plt.title('Mapa Causal — Cross-Correlation\n(arestas = lag em minutos)',
              color='#e6edf3', fontsize=9)
    out = os.path.join(output_dir, 'causal_map.png')
    plt.savefig(out, dpi=140, bbox_inches='tight',
                facecolor=plt.gcf().get_facecolor())
    plt.close()
    print(f"[CAUSAL] {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. TRANSFER ENTROPY
# ─────────────────────────────────────────────────────────────────────────────
def compute_transfer_entropy(df_enc, encoded_cols, output_dir,
                              max_lag=20, te_threshold=0.02):
    """
    TRANSFER ENTROPY (Schreiber, 2000):
      TE(X→Y) = H(Y_futuro | Y_passado) - H(Y_futuro | Y_passado, X_passado)

      Em linguagem simples:
        Quanto saber X_passado reduz a incerteza sobre Y_futuro,
        além do que Y_passado já explica.

      Unidade: bits
      TE = 0: X não acrescenta informação sobre Y
      TE > 0: X causa Y com informação adicional de TE bits

    DISCRETIZAÇÃO por quantis:
      pd.qcut(x, q=10) → divide a série em 10 bins de igual probabilidade
        (decis). Necessário porque Transfer Entropy é calculada sobre
        variáveis discretas (histogramas de probabilidade conjunta).
      q=10: resolução suficiente para capturar estrutura não-linear.
      duplicates='drop': remove bins duplicados quando a série é muito esparsa.

    DIFERENÇA PARA CROSS-CORRELATION:
      Cross-correlation: linear, simétrica, não distingue causa de efeito
      Transfer Entropy: não-linear, assimétrica, captura causalidade real
        TE(X→Y) ≠ TE(Y→X) em geral

    te_threshold = 0.02 bits:
      Limiar mínimo para reportar uma relação causal.
      Abaixo disso, a relação é estatisticamente marginal.

    REQUER: pip install pyinform
    """
    try:
        from pyinform.transferentropy import transfer_entropy
        from itertools import permutations
    except ImportError:
        print("[TE] pyinform não instalado — instala com: pip install pyinform")
        return pd.DataFrame()

    # Seleccionar apenas rate e latency (mais informativos para causalidade)
    cols = [c for c in df_enc.columns
            if c in encoded_cols and ('_rate' in c or '_latency' in c)]

    # Discretizar por decis — necessário para Transfer Entropy
    df_te = df_enc[cols].copy()
    for c in df_te.columns:
        df_te[c] = pd.qcut(df_te[c], q=10, labels=False, duplicates='drop')
    df_te = df_te.fillna(0).astype(int)

    def compute_te_lags(x, y, max_lag=20):
        """Calcula TE para múltiplos lags e retorna o máximo."""
        tes = []
        for lag in range(1, max_lag):
            try:
                te = transfer_entropy(x[:-lag], y[lag:], k=1)
                tes.append(te)
            except:
                tes.append(0)
        return max(tes)

    print("[TE] Calculando Transfer Entropy — pode demorar alguns minutos...")
    results = []
    from itertools import permutations
    for source, target in permutations(df_te.columns, 2):
        x = df_te[source].values
        y = df_te[target].values
        te = compute_te_lags(x, y, max_lag)
        if te > te_threshold:
            results.append({'source': source, 'target': target,
                            'transfer_entropy': te})

    te_df = pd.DataFrame(results).sort_values('transfer_entropy', ascending=False)
    print("\n[TE] Transfer Entropy detectada:")
    print(te_df.to_string())

    if not te_df.empty:
        out = os.path.join(output_dir, 'transfer_entropy.csv')
        te_df.to_csv(out, index=False)
        print(f"[TE] Guardado: {out}")

    return te_df

# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'outputs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Validar e reindexar ────────────────────────────────────────────
    df_val = load_and_validate(df)

    # ── 2. Detectar dead states ───────────────────────────────────────────
    df_val = build_dead_state_mask(df_val)

    # ── 3. Calcular os 30 encodings ───────────────────────────────────────
    df_enc, encoded_cols = encode_all(df_val)

    # ── 4. Visualização — 1 PNG por variável ─────────────────────────────
    plot_all(df_enc, output_dir=OUTPUT_DIR,
             window_hours=48, start_offset_hours=200)

    # ── 5. Métricas de qualidade das normalizações ────────────────────────
    df_metricas = metricas_normalizacao(df_val, df_enc, OUTPUT_DIR)

    # ── 6. Análise de falhas de sensor ────────────────────────────────────
    df_falhas = analise_falhas_sensor(df_val, df_enc, OUTPUT_DIR)

    # ── 7. Sazonalidade por mês ───────────────────────────────────────────
    analise_sazonal(df_val, OUTPUT_DIR)

    # ── 8. Mapa causal por cross-correlation ─────────────────────────────
    causal_df = build_causal_map(df_enc, encoded_cols, max_lag=60, min_corr=0.30)
    print("\n[CAUSAL MAP] Top 20 relações causais:")
    print(causal_df.head(20).to_string())
    causal_df.to_csv(os.path.join(OUTPUT_DIR, 'causal_edges.csv'), index=False)
    plot_causal_graph(causal_df, OUTPUT_DIR)

    # ── 9. Transfer Entropy (requer pyinform) ─────────────────────────────
    # Descomenta se tiveres pyinform instalado:
    # te_df = compute_transfer_entropy(df_enc, encoded_cols, OUTPUT_DIR)

    # ── 10. Tensor final para snnTorch ────────────────────────────────────
    X, indices = prepare_snn_tensor(df_enc, encoded_cols,
                                    window_steps=240, stride_steps=30)

    # Guardar tensor para usar no snn_model.py sem recalcular
    np.save(os.path.join(OUTPUT_DIR, 'X_tensor.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'indices.npy'), indices)

    print(f"\n{'='*60}")
    print(f"[PRONTO] X shape {X.shape}")
    print(f"[PRONTO] Tensor guardado em: {OUTPUT_DIR}/X_tensor.npy")
    print(f"[PRONTO] Próximo passo: snn_model.py")
    print(f"{'='*60}") 
    

    # ══════════════════════════════════════════════════════════════════════
    # PRÓXIMO SCRIPT — snn_model.py — O QUE VAI ACONTECER:
    #
    # import snntorch as snn
    # from snntorch import surrogate
    #
    # X = np.load('outputs/X_tensor.npy')          # (17512, 240, 30)
    # X_tensor = torch.FloatTensor(X).to(DEVICE)   # move para GPU
    #
    # # Arquitectura:
    # # Input:  30 canais (10 variáveis × 3 encodings)
    # # Hidden: 128 neurônios LIF com β = e^(-2/300) ≈ 0.9934
    # #         (τ_m = 5h = 300 steps × 2min → β calculado da física do processo)
    # # Output: 2 neurônios (PAO dominante / GAO dominante)
    # #
    # # Treino: BPTT com surrogate gradient ATan
    # # Loss:   ce_rate_loss (conta spikes) ou ce_temporal_loss (primeiro spike)
    # # Épocas: 50-100 com early stopping
    # ══════════════════════════════════════════════════════════════════════
