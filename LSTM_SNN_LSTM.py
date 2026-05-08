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

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen

# ==========================================
# CONFIGURAÇÕES
# ==========================================
num_steps  = 50
batch_size = 4
n_features = 17
n_neurons  = 64

# Dado sintético — shape: [batch, num_steps, features]
dado_continuo = torch.rand(batch_size, num_steps, n_features)

# ==========================================
# 1. SURROGATE GRADIENTS
# Posição: declarados ANTES dos neurônios
# Passados como argumento spike_grad
# Padrão da biblioteca se não especificado: ATan
# Fonte: https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html
# ==========================================

# ATan — padrão do snnTorch quando spike_grad não é especificado
# dS/dU ≈ α/2 / (1 + (α·π·U/2)²)
# Mais suave — gradiente viaja mais longe no tempo
spike_grad_atan = surrogate.atan(alpha=2.0)

# FastSigmoid — slope=25 é o padrão
# dS/dU ≈ slope / (slope·|U| + 1)²
# Mais íngreme — boa aproximação local
spike_grad_fast = surrogate.fast_sigmoid(slope=25)

# Sigmoid — similar ao FastSigmoid mas com formulação diferente
# dS/dU = k·exp(-kU) / (exp(-kU)+1)²
spike_grad_sigmoid = surrogate.sigmoid(slope=25)

# Straight Through Estimator — dS/dU = 1 em todo lugar
# Brutal mas funciona quando outros somem
spike_grad_ste = surrogate.straight_through_estimator()

print("=== SURROGATE GRADIENTS DISPONÍVEIS ===")
print("atan            — padrão da biblioteca, mais suave")
print("fast_sigmoid    — íngreme, boa aproximação local")
print("sigmoid         — variante do fast_sigmoid")
print("straight_through — dS/dU = 1 em todo lugar")

# ==========================================
# 2. ENCODING
# Posição: ANTES do loop temporal
# Converte dado contínuo em spikes
# Não faz parte do grafo — não recebe gradiente
# Fonte: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html
# ==========================================

# Um timestep de features para demonstração
x_sample = dado_continuo[:, 0, :]  # shape: [batch, features]

# --- Rate Coding ---
# magnitude → frequência de spikes
# P(S[t]=1) = x  onde x deve estar entre 0 e 1
# Shape saída: [num_steps, batch, features] — time-first
dado_rate = spikegen.rate(x_sample, num_steps=num_steps)
print(f"\n=== ENCODING ===")
print(f"Rate coding — shape: {dado_rate.shape}")
print(f"  taxa de spikes: {dado_rate.float().mean():.4f}")
print(f"  informação: magnitude → frequência de spikes")

# --- Latency Coding ---
# magnitude → quando o primeiro spike acontece
# tau: constante de tempo do encoding
# threshold: abaixo disso = sem spike (sensor morto)
# normalize=True: encaixa dentro de num_steps
# linear=False: encoding logarítmico (biologicamente mais preciso)
dado_latency = spikegen.latency(
    x_sample,
    num_steps=num_steps,
    tau=5,
    threshold=0.01,
    normalize=True,
    linear=False
)
print(f"\nLatency coding — shape: {dado_latency.shape}")
print(f"  taxa de spikes: {dado_latency.float().mean():.4f}")
print(f"  informação: magnitude → tempo do primeiro spike")

# --- Delta Modulation ---
# spike quando variação entre timesteps > threshold
# off_spike=True: spike negativo para quedas
# Biologicamente inspirado — neurônio responde à mudança
serie_temporal = dado_continuo[0, :, 0]  # uma feature ao longo do tempo
dado_delta = spikegen.delta(serie_temporal, threshold=0.1, off_spike=False)
print(f"\nDelta modulation — shape: {dado_delta.shape}")
print(f"  spikes gerados: {dado_delta.sum().int()}")
print(f"  informação: variação → spike de mudança")

# ==========================================
# 3. snn.Leaky — NEURÔNIO DE PRIMEIRA ORDEM
# Documentação oficial:
# https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html
#
# EQUAÇÕES OFICIAIS:
# subtract: U[t+1] = β·U[t] + I_in[t+1] - R·U_thr
# zero:     U[t+1] = β·U[t] + I_in[t+1] - R·(β·U[t] + I_in[t+1])
#
# beta   = membrane potential decay rate (decaimento do POTENCIAL)
# NÃO existe alpha no Leaky — corrente é instantânea
# ==========================================

fc1 = nn.Linear(n_features, n_neurons)

# Beta escalar — um único valor para toda a camada
# Todos os neurônios com mesmo decaimento
lif_escalar = snn.Leaky(
    beta=0.95,              # decay rate do potencial — entre 0 e 1
    threshold=1.0,          # U_thr padrão
    spike_grad=spike_grad_atan,  # surrogate gradient
    learn_beta=True,        # beta vira parâmetro treinável
    learn_threshold=True,   # threshold vira parâmetro treinável
    reset_mechanism="subtract"   # subtract ou zero
)

# Beta vetor — um valor por neurônio
# Calculado a partir do tempo físico: β = e^(-Δt/τ_real)
# Δt = 2min (frequência do sensor Agtrup)
beta_vetor = torch.cat([
    # Neurônios 0-15: τ = 2 steps = 4min (troca de fase rápida)
    torch.full((16,), torch.exp(torch.tensor(-2.0/4.0)).item()),   # β ≈ 0.607
    # Neurônios 16-31: τ = 15 steps = 30min (resposta do PO4)
    torch.full((16,), torch.exp(torch.tensor(-2.0/30.0)).item()),  # β ≈ 0.935
    # Neurônios 32-63: espectro livre 0.70 a 0.99
    torch.linspace(0.70, 0.99, 32)
])

lif_vetor = snn.Leaky(
    beta=beta_vetor,        # tensor [64] — um beta por neurônio
    threshold=1.0,
    spike_grad=spike_grad_atan,
    learn_beta=True,
    learn_threshold=True,
    reset_mechanism="subtract"
)

print(f"\n=== snn.Leaky ===")
print(f"Beta escalar shape: {lif_escalar.beta.shape}")
print(f"Beta vetor shape:   {lif_vetor.beta.shape}")
print(f"Equação: U[t+1] = β·U[t] + I_in[t+1] - R·U_thr")
print(f"Beta age no POTENCIAL — corrente é instantânea no Leaky")

# Forward pass demonstração
mem = lif_escalar.init_leaky()  # inicializa U[0] = 0
inp = dado_continuo[:, 0, :]
cur = fc1(inp)                  # corrente instantânea I[t] = W·X[t]
spk, mem = lif_escalar(cur, mem)  # aplica LIF — Heaviside + reset internos
print(f"cur shape: {cur.shape}  — corrente I[t]")
print(f"spk shape: {spk.shape}  — S[t] output do Heaviside")
print(f"mem shape: {mem.shape}  — U[t] potencial de membrana")
print(f"spikes ativos: {spk.sum().int()} de {spk.numel()}")

# ==========================================
# 4. snn.Synaptic — NEURÔNIO DE SEGUNDA ORDEM
# Documentação oficial:
# https://snntorch.readthedocs.io/en/latest/snn.neurons_synaptic.html
#
# EQUAÇÕES OFICIAIS:
# I_syn[t+1] = α·I_syn[t] + I_in[t+1]   ← alpha age NA CORRENTE
# U[t+1] = β·U[t] + I_syn[t+1] - R·U_thr ← beta age NO POTENCIAL
#
# alpha  = synaptic current decay rate (decaimento da CORRENTE)
# beta   = membrane potential decay rate (decaimento do POTENCIAL)
# São dois parâmetros SEPARADOS para dois estados SEPARADOS
# ==========================================

lif_synaptic = snn.Synaptic(
    alpha=0.90,             # decaimento da corrente sináptica I_syn
    beta=0.80,              # decaimento do potencial de membrana U
    threshold=1.0,
    spike_grad=spike_grad_atan,
    learn_alpha=True,       # alpha vira parâmetro treinável
    learn_beta=True,        # beta vira parâmetro treinável
    learn_threshold=True,
    reset_mechanism="subtract"
)

syn, mem_syn = lif_synaptic.init_synaptic()  # inicializa I_syn[0]=0, U[0]=0
spk_syn, syn, mem_syn = lif_synaptic(cur, syn, mem_syn)

print(f"\n=== snn.Synaptic ===")
print(f"Equação corrente:  I_syn[t+1] = α·I_syn[t] + I_in[t+1]")
print(f"Equação potencial: U[t+1] = β·U[t] + I_syn[t+1] - R·U_thr")
print(f"alpha age na CORRENTE — beta age no POTENCIAL — são independentes")
print(f"syn (corrente) shape: {syn.shape}")
print(f"mem (potencial) shape: {mem_syn.shape}")

# ==========================================
# 5. snn.RLeaky — NEURÔNIO RECORRENTE
# Documentação oficial:
# https://snntorch.readthedocs.io/en/latest/snn.neurons_rleaky.html
#
# EQUAÇÃO OFICIAL:
# U[t+1] = β·U[t] + I_in[t+1] + V(S_out[t]) - R·U_thr
#
# V = peso recorrente — escala o spike de saída realimentado
# ==========================================

lif_recorrente = snn.RLeaky(
    beta=0.85,
    linear_features=n_neurons,  # dimensão para conexão recorrente all-to-all
    threshold=1.0,
    spike_grad=spike_grad_atan,
    learn_beta=True,
    learn_threshold=True,
    learn_recurrent=True,        # pesos recorrentes V são treináveis
    reset_mechanism="subtract"
)

spk_r, mem_r = lif_recorrente.init_rleaky()
fc_r = nn.Linear(n_features, n_neurons)
cur_r = fc_r(inp)
spk_r, mem_r = lif_recorrente(cur_r, spk_r, mem_r)

print(f"\n=== snn.RLeaky ===")
print(f"Equação: U[t+1] = β·U[t] + I_in[t+1] + V(S_out[t]) - R·U_thr")
print(f"V = peso recorrente que realimenta o spike de saída")

# ==========================================
# 6. RESET MECHANISMS
# Documentação oficial — dois modos:
# subtract: U[t+1] = β·U[t] + I[t+1] - R·U_thr
# zero:     U[t+1] = β·U[t] + I[t+1] - R·(β·U[t] + I[t+1])
# ==========================================

lif_subtract = snn.Leaky(beta=0.80, reset_mechanism="subtract",
                          spike_grad=spike_grad_atan)
lif_zero     = snn.Leaky(beta=0.80, reset_mechanism="zero",
                          spike_grad=spike_grad_atan)

mem_sub = lif_subtract.init_leaky()
mem_zer = lif_zero.init_leaky()

print(f"\n=== RESET MECHANISMS ===")
fc_reset = nn.Linear(n_features, n_neurons)
for t in range(3):
    cur_t = fc_reset(inp) * 2.0  # corrente alta para forçar spikes
    spk_s, mem_sub = lif_subtract(cur_t, mem_sub)
    spk_z, mem_zer = lif_zero(cur_t, mem_zer)
    print(f"t={t} | subtract mem: {mem_sub.mean().item():+.4f} "
          f"spk: {spk_s.sum().int()} | "
          f"zero mem: {mem_zer.mean().item():+.4f} "
          f"spk: {spk_z.sum().int()}")

# ==========================================
# 7. REDE COMPLETA — TODOS OS COMPONENTES
# Cronograma do fluxo conforme documentação oficial
# ==========================================

class SNN_Completo(nn.Module):
    """
    CRONOGRAMA DO FLUXO:

    [1] Input x[:, step, :]
        shape: [batch, features]

    [2] Encoding — ANTES do loop
        rate / latency / delta
        converte dado contínuo em spikes

    [3] Loop temporal for step in range(num_steps):

        [3a] Corrente I[t] — nn.Linear
             cur = W · X[t]
             shape: [batch, neurons]
             No Leaky: corrente instantânea, sem memória
             No Synaptic: I_syn acumula via alpha

        [3b] Potencial U[t] — snn.Leaky / snn.Synaptic
             Leaky:    U[t+1] = β·U[t] + I_in[t+1]
             Synaptic: U[t+1] = β·U[t] + I_syn[t+1]
             β age aqui — no potencial
             mem É o U[t] no código

        [3c] Threshold U_thr
             valor de referência passivo
             critério que o Heaviside usa

        [3d] Heaviside Θ — DENTRO do lif
             S[t] = Θ(U[t] - U_thr)
             decide 0 E 1 — não só o 0
             não é parâmetro — é função de decisão

        [3e] Reset R[t] — DENTRO do lif, após spike
             subtract: preserva excesso
             zero: apaga potencial

        [3f] Spike S[t] propaga para próxima camada
             mem persiste para próximo timestep

    [4] Loss em mem[-1]
        potencial do último timestep
        carrega história acumulada

    [5] Surrogate — só no backward
        substitui dΘ/dU
        não muda o forward
    """

    def __init__(self, inputs, neurons=64):
        super().__init__()

        # ---- Camada 1: snn.Leaky com beta vetor ----
        # β calculado a partir do tempo físico real
        # β = e^(-Δt/τ_real) onde Δt=2min
        beta_c1 = torch.cat([
            torch.full((neurons//2,),
                       torch.exp(torch.tensor(-2.0/4.0)).item()),   # τ=4min
            torch.linspace(0.80, 0.99, neurons//2)                  # lento
        ])
        self.fc1  = nn.Linear(inputs, neurons)
        self.lif1 = snn.Leaky(
            beta=beta_c1,           # vetor — um β por neurônio
            threshold=1.0,
            spike_grad=surrogate.atan(alpha=2.0),  # padrão snnTorch
            learn_beta=True,
            learn_threshold=True,
            reset_mechanism="subtract"             # preserva excesso
        )

        # ---- Camada 2: snn.Synaptic com alpha e beta ----
        # alpha: decaimento da corrente sináptica
        # beta: decaimento do potencial de membrana
        # São INDEPENDENTES — não se misturam
        self.fc2  = nn.Linear(neurons, neurons)
        self.lif2 = snn.Synaptic(
            alpha=0.90,             # decay da CORRENTE I_syn[t]
            beta=0.70,              # decay do POTENCIAL U[t]
            threshold=1.0,
            spike_grad=surrogate.fast_sigmoid(slope=25),
            learn_alpha=True,
            learn_beta=True,
            learn_threshold=True,
            reset_mechanism="subtract"
        )

        # ---- Camada 3: snn.RLeaky com realimentação ----
        self.fc3  = nn.Linear(neurons, neurons // 2)
        self.lif3 = snn.RLeaky(
            beta=0.85,
            linear_features=neurons // 2,
            threshold=1.0,
            spike_grad=surrogate.atan(alpha=2.0),
            learn_beta=True,
            learn_recurrent=True,
            reset_mechanism="zero"                 # apaga após spike
        )

        # ---- Saída: regressão contínua ----
        self.fc_out = nn.Linear(neurons // 2, 1)

    def forward(self, x):
        """
        x shape: [batch, num_steps, features]
        """
        # Inicializa estados — U[0] = 0
        mem1               = self.lif1.init_leaky()
        syn2, mem2         = self.lif2.init_synaptic()
        spk3, mem3         = self.lif3.init_rleaky()

        spk1_rec = []
        spk2_rec = []
        mem3_rec = []

        for step in range(num_steps):
            inp_t = x[:, step, :]  # [batch, features] — timestep atual

            # [3a] Corrente instantânea no Leaky
            # I[t] = W · X[t] — sem alpha, sem memória
            cur1 = self.fc1(inp_t)

            # [3b-3e] Potencial + Threshold + Heaviside + Reset
            # U[t+1] = β·U[t] + cur1 - R·U_thr  (subtract)
            # β age no potencial — corrente é instantânea
            spk1, mem1 = self.lif1(cur1, mem1)

            # [3a] Corrente com dinâmica no Synaptic
            # I_syn[t+1] = α·I_syn[t] + W·spk1  ← alpha age aqui
            cur2 = self.fc2(spk1)

            # [3b-3e]
            # U[t+1] = β·U[t] + I_syn[t+1] - R·U_thr ← beta age aqui
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

            # [3a-3e] RLeaky com realimentação
            # U[t+1] = β·U[t] + I_in[t+1] + V(S_out[t]) - R·(β·U+I)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, spk3, mem3)

            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            mem3_rec.append(mem3)

        # [4] Saída — potencial do último timestep
        # mem3_rec[-1] carrega história acumulada de todos os timesteps
        output = self.fc_out(mem3_rec[-1])

        return (
            output,
            torch.stack(spk1_rec),   # [num_steps, batch, neurons]
            torch.stack(spk2_rec),   # [num_steps, batch, neurons]
            torch.stack(mem3_rec)    # [num_steps, batch, neurons//2]
        )


# ==========================================
# 8. EXECUÇÃO E INSPEÇÃO
# ==========================================
net = SNN_Completo(inputs=n_features, neurons=n_neurons)

output, spk1, spk2, mem3 = net(dado_continuo)

print("\n" + "="*55)
print("INSPEÇÃO DA REDE")
print("="*55)

print(f"\nOutput (regressão): {output.shape}")
print(f"Spikes camada 1 (Leaky):    {spk1.shape}")
print(f"Spikes camada 2 (Synaptic): {spk2.shape}")
print(f"Potencial camada 3 (RLeaky):{mem3.shape}")

print(f"\n--- Taxa de spikes por camada ---")
print(f"Leaky   (beta vetor 0.607-0.99): {spk1.float().mean():.4f}")
print(f"Synaptic (alpha=0.90, beta=0.70): {spk2.float().mean():.4f}")

print(f"\n--- Parâmetros aprendíveis ---")
# Camada 1 — Leaky — beta vetor
beta1 = net.lif1.beta.detach()
tau1  = -1 / torch.log(beta1.clamp(min=1e-6))
print(f"Leaky beta vetor:")
print(f"  Neurônios 0-31  (rápidos): β={beta1[:32].mean():.4f} "
      f"→ τ={tau1[:32].mean():.1f} steps")
print(f"  Neurônios 32-63 (lentos):  β={beta1[32:].mean():.4f} "
      f"→ τ={tau1[32:].mean():.1f} steps")
print(f"  Threshold: {net.lif1.threshold.item():.4f}")

# Camada 2 — Synaptic — alpha E beta separados
print(f"\nSynaptic:")
print(f"  alpha (corrente):   {net.lif2.alpha.item():.4f}")
print(f"  beta  (potencial):  {net.lif2.beta.item():.4f} "
      f"→ τ={-1/torch.log(net.lif2.beta.clamp(min=1e-6)).item():.1f} steps")
print(f"  Threshold: {net.lif2.threshold.item():.4f}")

print(f"\n--- Surrogate gradients usados ---")
print(f"Camada 1 (Leaky):    atan(alpha=2.0)      — padrão snnTorch")
print(f"Camada 2 (Synaptic): fast_sigmoid(slope=25)")
print(f"Camada 3 (RLeaky):   atan(alpha=2.0)")

print(f"\n--- Reset mechanisms usados ---")
print(f"Camada 1: subtract — U[t+1] = β·U[t] + I[t+1] - R·U_thr")
print(f"Camada 2: subtract — preserva excesso após spike")
print(f"Camada 3: zero     — U[t+1] = β·U[t] + I[t+1] - R·(β·U[t]+I[t+1])")

print(f"\n--- Parâmetros totais ---")
total      = sum(p.numel() for p in net.parameters())
betas      = sum(p.numel() for n, p in net.named_parameters() if 'beta' in n)
alphas     = sum(p.numel() for n, p in net.named_parameters() if 'alpha' in n)
thresholds = sum(p.numel() for n, p in net.named_parameters() if 'threshold' in n)
pesos      = total - betas - alphas - thresholds
print(f"Total:      {total}")
print(f"Pesos W:    {pesos}")
print(f"Betas β:    {betas}")
print(f"Alphas α:   {alphas}")
print(f"Thresholds: {thresholds}")

print("\n" + "="*55)
print("RESUMO DO QUE FOI CORRIGIDO")
print("="*55)
print("""
1. snn.Leaky:
   - beta age NO POTENCIAL — não na corrente
   - corrente é instantânea: I[t] = W·X[t]
   - NÃO existe alpha no Leaky

2. snn.Synaptic:
   - alpha age NA CORRENTE: I_syn[t+1] = α·I_syn[t] + I_in[t+1]
   - beta age NO POTENCIAL: U[t+1] = β·U[t] + I_syn[t+1]
   - são dois parâmetros SEPARADOS para dois estados SEPARADOS

3. Heaviside:
   - decide os dois lados — 0 E 1
   - não é só o 0

4. Surrogate padrão da biblioteca:
   - ATan (não fast_sigmoid) quando spike_grad não é especificado

5. Reset:
   - subtract (oficial): U[t+1] = β·U[t] + I[t+1] - R·U_thr
   - zero (oficial):     U[t+1] = β·U[t] + I[t+1] - R·(β·U[t]+I[t+1])
""")

