import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.fftpack import fft
import pywt
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import os
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
//teste gitt
print("--- SISTEMA DE PROTEÇÃO: LOCALIZADOR E CLASSIFICADOR DE FALTAS (WAVELET DB4) ---")

# ================= SELEÇÃO DO ARQUIVO =================
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)

caminho_selecionado = filedialog.askopenfilename(
    title="Selecione o arquivo .ADF exportado do PlotXY",
    filetypes=[("Arquivo de Texto ATP", "*.adf"), ("Todos os arquivos", "*.*")]
)
root.destroy()

if not caminho_selecionado:
    print("Seleção cancelada pelo usuário.")
    exit()

ARQUIVO_ADF = caminho_selecionado
print(f"\n[*] Processando simulação: {os.path.basename(ARQUIVO_ADF)}")

VELOCIDADE_PROP = 295000.0  # km/s

# ================= LEITURA DO ADF =================
def carregar_adf(caminho):
    try:
        with open(caminho, 'r') as f:
            linhas = f.readlines()
        dados = []
        for linha in linhas:
            partes = linha.split()
            try:
                valores = [float(p) for p in partes]
                if len(valores) >= 4:
                    dados.append(valores[:4])
            except ValueError:
                continue
        if not dados:
            return None, "Sem dados numéricos."
        df = pd.DataFrame(dados, columns=['t', 'VA', 'VB', 'VC'])
        df = df.sort_values(by='t').reset_index(drop=True)
        return df, "Sucesso"
    except Exception as e:
        return None, str(e)

# ================= COMPONENTES SIMÉTRICAS =================
def calcular_componentes_simetricas(Va, Vb, Vc):
    """
    Calcula componentes de sequência zero, positiva e negativa
    """
    a = np.exp(2j * np.pi / 3)
    
    # Matriz de transformação
    T = np.array([
        [1, 1, 1],
        [1, a, a**2],
        [1, a**2, a]
    ]) / 3
    
    # Componentes de sequência
    V_seq = T @ np.array([Va, Vb, Vc])
    
    V0 = V_seq[0]  # Sequência zero
    V1 = V_seq[1]  # Sequência positiva
    V2 = V_seq[2]  # Sequência negativa
    
    return V0, V1, V2

# ================= EXTRAÇÃO DE WAVELET =================
def extract_wavelet_detail(signal):
    tamanho_original = len(signal)
    if tamanho_original % 2 != 0:
        signal = np.append(signal, signal[-1])
    coeffs = pywt.swt(signal, 'db4', level=1)
    cA, cD = coeffs[0]
    return np.abs(cD[:tamanho_original])

# ================= EXTRAÇÃO DE CARACTERÍSTICAS =================
def extrair_caracteristicas(Va, Vb, Vc, dt):
    """
    Extrai características para classificação de faltas
    """
    caracteristicas = {}
    
    # 1. Componentes Simétricas (rms e pico)
    V0_seq, V1_seq, V2_seq = calcular_componentes_simetricas(Va, Vb, Vc)
    
    caracteristicas['V0_rms'] = np.sqrt(np.mean(np.abs(V0_seq)**2))
    caracteristicas['V1_rms'] = np.sqrt(np.mean(np.abs(V1_seq)**2))
    caracteristicas['V2_rms'] = np.sqrt(np.mean(np.abs(V2_seq)**2))
    
    # Razões de sequência (indicadores importantes)
    caracteristicas['V2_V1_ratio'] = caracteristicas['V2_rms'] / (caracteristicas['V1_rms'] + 1e-10)
    caracteristicas['V0_V1_ratio'] = caracteristicas['V0_rms'] / (caracteristicas['V1_rms'] + 1e-10)
    
    # 2. Razões entre fases (voltagem e corrente)
    Va_rms = np.sqrt(np.mean(Va**2))
    Vb_rms = np.sqrt(np.mean(Vb**2))
    Vc_rms = np.sqrt(np.mean(Vc**2))
    
    caracteristicas['Va_rms'] = Va_rms
    caracteristicas['Vb_rms'] = Vb_rms
    caracteristicas['Vc_rms'] = Vc_rms
    
    # Razões entre fases
    caracteristicas['Va_Vb_ratio'] = Va_rms / (Vb_rms + 1e-10)
    caracteristicas['Va_Vc_ratio'] = Va_rms / (Vc_rms + 1e-10)
    caracteristicas['Vb_Vc_ratio'] = Vb_rms / (Vc_rms + 1e-10)
    
    # 3. Desvio padrão entre fases
    caracteristicas['std_fases'] = np.std([Va_rms, Vb_rms, Vc_rms])
    
    # 4. Wavelet energy
    cD_a = extract_wavelet_detail(Va)
    cD_b = extract_wavelet_detail(Vb)
    cD_c = extract_wavelet_detail(Vc)
    
    caracteristicas['wavelet_a'] = np.mean(cD_a**2)
    caracteristicas['wavelet_b'] = np.mean(cD_b**2)
    caracteristicas['wavelet_c'] = np.mean(cD_c**2)
    
    # 5. THD (Total Harmonic Distortion) simplificado
    fft_a = np.abs(fft(Va))
    fft_b = np.abs(fft(Vb))
    fft_c = np.abs(fft(Vc))
    
    fundamental_idx = np.argmax(fft_a)
    caracteristicas['thd_a'] = np.sqrt(np.sum(fft_a[fundamental_idx+1:]**2)) / fft_a[fundamental_idx]
    
    # 6. Ângulo de fase entre fases
    Va_angle = np.arctan2(np.imag(fft_a[fundamental_idx]), np.real(fft_a[fundamental_idx]))
    Vb_angle = np.arctan2(np.imag(fft_b[fundamental_idx]), np.real(fft_b[fundamental_idx]))
    Vc_angle = np.arctan2(np.imag(fft_c[fundamental_idx]), np.real(fft_c[fundamental_idx]))
    
    caracteristicas['angle_ab'] = np.abs(Va_angle - Vb_angle)
    caracteristicas['angle_ac'] = np.abs(Va_angle - Vc_angle)
    caracteristicas['angle_bc'] = np.abs(Vb_angle - Vc_angle)
    
    # 7. Assimetria
    caracteristicas['skewness_a'] = np.mean((Va - np.mean(Va))**3) / (np.std(Va)**3 + 1e-10)
    caracteristicas['skewness_b'] = np.mean((Vb - np.mean(Vb))**3) / (np.std(Vb)**3 + 1e-10)
    caracteristicas['skewness_c'] = np.mean((Vc - np.mean(Vc))**3) / (np.std(Vc)**3 + 1e-10)
    
    return caracteristicas

# ================= CLASSIFICAÇÃO DE FALTAS =================
def classificar_falta(caracteristicas):
    """
    Classifica o tipo de falta baseado nas características extraídas
    Tipos: AN, ABN, ACN, AB, ABC, ABG, ACG, ABG, ABCG
    """
    
    V0 = caracteristicas['V0_rms']
    V1 = caracteristicas['V1_rms']
    V2 = caracteristicas['V2_rms']
    
    Va_rms = caracteristicas['Va_rms']
    Vb_rms = caracteristicas['Vb_rms']
    Vc_rms = caracteristicas['Vc_rms']
    
    V2_V1 = caracteristicas['V2_V1_ratio']
    V0_V1 = caracteristicas['V0_V1_ratio']
    
    std_fases = caracteristicas['std_fases']
    
    # Classificação baseada em lógica fuzzy e limites
    
    # Falta trifásica (ABC ou ABCG)
    if std_fases < 0.1 and V2_V1 < 0.15 and V0_V1 < 0.2:
        if V0 > V1 * 0.05:
            return "ABCG (Trifásica-Terra)", 0.95
        else:
            return "ABC (Trifásica)", 0.95
    
    # Falta bifásica (AB, AC, BC)
    if V2_V1 > 0.3 and V2_V1 < 0.9:
        min_v = min(Va_rms, Vb_rms, Vc_rms)
        max_v = max(Va_rms, Vb_rms, Vc_rms)
        
        if min_v / (max_v + 1e-10) > 0.7:  # Duas fases similares
            # Verificar quais fases caíram
            if Va_rms < Vb_rms and Va_rms < Vc_rms:
                if abs(Vb_rms - Vc_rms) < max_v * 0.1:
                    if V0 > V1 * 0.05:
                        return "AG (Fase A-Terra)", 0.85
                    else:
                        return "A (Fase A)", 0.85
            elif Vb_rms < Va_rms and Vb_rms < Vc_rms:
                if abs(Va_rms - Vc_rms) < max_v * 0.1:
                    if V0 > V1 * 0.05:
                        return "BG (Fase B-Terra)", 0.85
                    else:
                        return "B (Fase B)", 0.85
            elif Vc_rms < Va_rms and Vc_rms < Vb_rms:
                if abs(Va_rms - Vb_rms) < max_v * 0.1:
                    if V0 > V1 * 0.05:
                        return "CG (Fase C-Terra)", 0.85
                    else:
                        return "C (Fase C)", 0.85
        
        # Bifásica
        elif Va_rms < max_v * 0.3 or Vb_rms < max_v * 0.3 or Vc_rms < max_v * 0.3:
            if Va_rms < Vb_rms * 0.3 and Va_rms < Vc_rms * 0.3:
                if V0 > V1 * 0.05:
                    return "ABG (Bifásica-Terra)", 0.80
                else:
                    return "AB (Bifásica)", 0.80
            elif Vb_rms < Va_rms * 0.3 and Vb_rms < Vc_rms * 0.3:
                if V0 > V1 * 0.05:
                    return "BCG (Bifásica-Terra)", 0.80
                else:
                    return "BC (Bifásica)", 0.80
            elif Vc_rms < Va_rms * 0.3 and Vc_rms < Vb_rms * 0.3:
                if V0 > V1 * 0.05:
                    return "ACG (Bifásica-Terra)", 0.80
                else:
                    return "AC (Bifásica)", 0.80
    
    # Falta monofásica
    if V2_V1 < 0.5 and V0_V1 > 0.3:
        if Va_rms < Vb_rms * 0.5 and Va_rms < Vc_rms * 0.5:
            return "AG (Fase A-Terra)", 0.88
        elif Vb_rms < Va_rms * 0.5 and Vb_rms < Vc_rms * 0.5:
            return "BG (Fase B-Terra)", 0.88
        elif Vc_rms < Va_rms * 0.5 and Vc_rms < Vb_rms * 0.5:
            return "CG (Fase C-Terra)", 0.88
    
    return "Desconhecida", 0.5

# ================= EXECUÇÃO PRINCIPAL =================
df, msg = carregar_adf(ARQUIVO_ADF)

if df is not None:
    try:
        dt = df['t'].values[1] - df['t'].values[0]
        VA, VB, VC = df['VA'].values, df['VB'].values, df['VC'].values
        
        # === LOCALIZADOR WAVELET ===
        V_0 = (VA + VB + VC) / 3.0
        V_alpha = (2.0/3.0) * (VA - 0.5*VB - 0.5*VC)
        V_beta = (2.0/3.0) * ((np.sqrt(3)/2.0)*VB - (np.sqrt(3)/2.0)*VC)
        
        cD_alpha = extract_wavelet_detail(V_alpha)
        cD_beta = extract_wavelet_detail(V_beta)
        cD_zero = extract_wavelet_detail(V_0)
        
        sinal_rele_tw = (cD_alpha**2) + (cD_beta**2) + (cD_zero**2)
        limiar = np.max(sinal_rele_tw) * 0.02
        
        picos_idx, _ = find_peaks(sinal_rele_tw, height=limiar, distance=5)

        t1, t2 = None, None
        idx_t1, idx_t2 = None, None
        distancia = 0.0
        
        if len(picos_idx) > 0:
            idx_t1 = picos_idx[0]
            t1 = df['t'].values[idx_t1]
            
            for idx in picos_idx[1:]:
                t_atual = df['t'].values[idx]
                if t_atual - t1 >= 45e-6:
                    t2 = t_atual
                    idx_t2 = idx
                    break

        # === CLASSIFICADOR DE FALTAS ===
        # Usar janela ao redor da falta para melhor classificação
        janela_inicio = max(0, idx_t1 - 100) if idx_t1 else 0
        janela_fim = min(len(VA), idx_t1 + 500) if idx_t1 else len(VA)
        
        VA_janela = VA[janela_inicio:janela_fim]
        VB_janela = VB[janela_inicio:janela_fim]
        VC_janela = VC[janela_inicio:janela_fim]
        
        caracteristicas = extrair_caracteristicas(VA_janela, VB_janela, VC_janela, dt)
        tipo_falta, confianca = classificar_falta(caracteristicas)

        print("="*70)
        if t1 is not None and t2 is not None:
            distancia = (VELOCIDADE_PROP * (t2 - t1)) / 2.0
            print(" RESULTADO DA ANÁLISE:")
            print(f" -> Tempo Incidente (T1):     {t1*1000:.4f} ms")
            print(f" -> Tempo Refletido (T2):     {t2*1000:.4f} ms")
            print(f" -> Delta T (T2 - T1):        {(t2-t1)*1000:.4f} ms")
            print(f" -> DISTÂNCIA CALCULADA:      {distancia:.3f} km")
        else:
            print(" [!] ERRO DE LOCALIZAÇÃO: O T2 não foi encontrado.")
        
        print("-" * 70)
        print(f" CLASSIFICAÇÃO DA FALTA:")
        print(f" -> Tipo: {tipo_falta}")
        print(f" -> Confiança: {confianca*100:.1f}%")
        print("="*70)
        
        # Exibir características extraídas
        print("\n CARACTERÍSTICAS EXTRAÍDAS:")
        for chave, valor in caracteristicas.items():
            print(f"  {chave}: {valor:.4f}")

    except Exception as e:
        print("\n[X] OCORREU UM ERRO FATAL DURANTE OS CÁLCULOS:")
        traceback.print_exc()
    
    # ================= GRÁFICOS =================
    try:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        titulo_dist = f"{distancia:.2f} km - {tipo_falta}" if t1 is not None and t2 is not None else f"Erro de Reflexo - {tipo_falta}"
        
        # Tensões
        axes[0].set_title(f"Tensão Instantânea - {titulo_dist}")
        axes[0].plot(df['t'], VA, color='yellow', label='Fase A', linewidth=1)
        axes[0].plot(df['t'], VB, color='cyan', label='Fase B', linewidth=1)
        axes[0].plot(df['t'], VC, color='magenta', label='Fase C', linewidth=1)
        if t1 is not None:
            axes[0].axvline(x=t1, color='red', linestyle='--', alpha=0.7, label='T1 (Incidente)')
        if t2 is not None:
            axes[0].axvline(x=t2, color='orange', linestyle='--', alpha=0.7, label='T2 (Refletido)')
        axes[0].set_ylabel("Tensão (V)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Wavelet
        axes[1].set_title("Coeficientes Wavelet DB4 (Energia)")
        axes[1].plot(df['t'], sinal_rele_tw, color='lime', linewidth=1.5)
        if t1 is not None:
            axes[1].plot(t1, sinal_rele_tw[idx_t1], 'ro', markersize=10, label='T1')
        if t2 is not None:
            axes[1].plot(t2, sinal_rele_tw[idx_t2], 'ro', markersize=10, label='T2')
        axes[1].set_ylabel("Energia")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Componentes Simétricas
        V0_seq, V1_seq, V2_seq = calcular_componentes_simetricas(VA, VB, VC)
        axes[2].set_title("Componentes Simétricas (RMS)")
        axes[2].plot(df['t'], np.abs(V0_seq), color='red', label='V0 (Zero)', linewidth=1)
        axes[2].plot(df['t'], np.abs(V1_seq), color='green', label='V1 (Positiva)', linewidth=1)
        axes[2].plot(df['t'], np.abs(V2_seq), color='blue', label='V2 (Negativa)', linewidth=1)
        axes[2].set_xlabel("Tempo (s)")
        axes[2].set_ylabel("Tensão (V)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        if t1 is not None:
            zoom_inicio = max(0, t1 - 0.002)
            zoom_fim = min(df['t'].values.max(), t2 + 0.005) if t2 is not None else t1 + 0.01
            for ax in axes:
                ax.set_xlim(zoom_inicio, zoom_fim)
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Erro ao gerar gráficos: {e}")

else:
    print(f"\n[X] FALHA AO LER ARQUIVO:\n{msg}")