"""
analisis_act_id_raw.py

Ultimo chequeo del intento de Sussillo-Barak para el Paper 3. Usa
act_id_raw (la decision CRUDA de la red, capturada por
collect_h_x_actraw_vectors.py, ANTES de que el instinto de auxilio la
pueda pisar) para responder dos preguntas, sin ningun ruido de
heuristicas externas de por medio:

  1. Frecuencia: durante el atasco, ¿con que frecuencia elige cada
     agente girar a la derecha (accion 2)? ¿Es distinto entre A y B?

  2. Persistencia ("giro sostenido"): dado que el agente ELIGIO
     girar a la derecha en un paso, ¿que probabilidad hay de que
     elija lo mismo en el paso siguiente? Un numero alto es la firma
     comportamental de lo que el Paper 2 describe como "sustained
     one-way turns" -- y es tambien la contraparte comportamental
     directa de la estabilidad local que encontramos con
     Sussillo-Barak (un punto localmente estable, por definicion, es
     un punto al que la dinamica vuelve solo -- eso se deberia ver
     como "elige lo mismo de nuevo").

Uso:
    python3 analisis_act_id_raw.py --csv h_x_actraw_300.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

ACTION_NAMES = {0: "adelante", 1: "giro_izq", 2: "giro_der", 3: "reversa", 4: "señalizar/wander"}
ACCION_DE_INTERES = 2  # giro_der -- la que aparecia en el sesgo de pesos del Paper 2


def frecuencia_por_condicion(df):
    print("=== Frecuencia de act_id_raw (decision cruda, sin el instinto) ===\n")
    for condicion, filtro_stalled in [("TODAS las decisiones", None), ("SOLO durante stalled=True", True)]:
        print(f"--- {condicion} ---")
        for agente in ["a", "b"]:
            if filtro_stalled is None:
                df_ag = df[df["agent"] == agente]
            else:
                df_ag = df[(df["agent"] == agente) & (df["stalled"] == filtro_stalled)]
            conteo = df_ag["act_id_raw"].value_counts(normalize=True).sort_index()
            etiquetas = {ACTION_NAMES.get(k, k): f"{v*100:.1f}%" for k, v in conteo.items()}
            print(f"  agente {agente} (n={len(df_ag)}): {etiquetas}")
        print()


def test_diferencia_giro_der_durante_atasco(df):
    print("=== Test estadistico: frecuencia de giro_der durante el atasco, A vs B ===\n")
    df_a = df[(df["agent"] == "a") & (df["stalled"] == True)]
    df_b = df[(df["agent"] == "b") & (df["stalled"] == True)]

    giro_der_a = (df_a["act_id_raw"] == ACCION_DE_INTERES).sum()
    giro_der_b = (df_b["act_id_raw"] == ACCION_DE_INTERES).sum()
    n_a, n_b = len(df_a), len(df_b)
    p_a, p_b = giro_der_a / n_a, giro_der_b / n_b

    tabla = np.array([[giro_der_a, n_a - giro_der_a], [giro_der_b, n_b - giro_der_b]])
    chi2, p_valor, dof, esperado = stats.chi2_contingency(tabla)

    print(f"  Agente A: giro_der en {giro_der_a}/{n_a} decisiones durante atasco = {p_a*100:.1f}%")
    print(f"  Agente B: giro_der en {giro_der_b}/{n_b} decisiones durante atasco = {p_b*100:.1f}%")
    print(f"  Diferencia: {(p_b-p_a)*100:.1f} puntos porcentuales")
    print(f"  Chi-cuadrado: chi2={chi2:.1f}, p={p_valor:.2e}\n")


def persistencia_giro_sostenido(df):
    print("=== Persistencia: dado que el paso anterior fue giro_der, "
          "¿que probabilidad hay de repetirlo? ===\n")
    df = df.sort_values(["agent", "episode", "decision"]).reset_index(drop=True)

    for agente in ["a", "b"]:
        df_ag = df[(df["agent"] == agente) & (df["stalled"] == True)].reset_index(drop=True)

        es_giro_der = df_ag["act_id_raw"] == ACCION_DE_INTERES
        giro_der_anterior = es_giro_der.shift(1).fillna(False)
        repite = es_giro_der & giro_der_anterior

        n_base = giro_der_anterior.sum()
        prob_persistencia = repite.sum() / max(n_base, 1)

        print(f"  agente {agente} (n={n_base} transiciones desde giro_der): "
              f"probabilidad de repetir = {prob_persistencia*100:.1f}%")

    print("\n  Numero alto = 'giro sostenido' (firma de 'sustained one-way turns' "
          "del Paper 2). Numero bajo = el agente cambia de accion seguido, "
          "consistente con una dinamica local inestable que lo 'suelta'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="h_x_actraw_300.csv",
                     help="CSV generado por collect_h_x_actraw_vectors.py (necesita la columna act_id_raw)")
    args = ap.parse_args()

    print(f"Leyendo {args.csv}...\n")
    df = pd.read_csv(args.csv)

    if "act_id_raw" not in df.columns:
        raise ValueError(
            f"Falta la columna act_id_raw en {args.csv} -- este script necesita "
            f"un CSV generado con collect_h_x_actraw_vectors.py (no collect_h_x_vectors.py, "
            f"que no guarda la decision cruda de la red)."
        )

    frecuencia_por_condicion(df)
    test_diferencia_giro_der_durante_atasco(df)
    persistencia_giro_sostenido(df)


if __name__ == "__main__":
    main()
