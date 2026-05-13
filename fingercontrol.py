"""
╔══════════════════════════════════════════════════════╗
║         FINGER MOUSE - Control tu PC con la mano     ║
║         Autor: Alex (villaalextor)                   ║
║         Deps: opencv-python, mediapipe, pyautogui    ║
╚══════════════════════════════════════════════════════╝

GESTOS:
  ☝️  Índice arriba          → Mover cursor
  🤏 Pellizco (índice+pulgar) → Clic izquierdo
  ✌️  Índice + medio arriba  → Modo scroll (mover arriba/abajo)
  ✊  Puño cerrado           → Pausar/Reanudar
  🖐️  Mano abierta (5 dedos) → Clic derecho
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ─── Configuración ──────────────────────────────────────────
pyautogui.FAILSAFE = False   # Desactiva la zona de emergencia en esquina
pyautogui.PAUSE = 0          # Sin pausa entre acciones (más fluido)

CAMARA_ID       = 0          # 0 = cámara principal de la laptop
SUAVIZADO       = 7          # Frames para promediar (más = más suave)
SENSIBILIDAD_X  = 1.8        # Multiplicador de movimiento horizontal
SENSIBILIDAD_Y  = 1.8        # Multiplicador de movimiento vertical
UMBRAL_CLICK    = 40         # Distancia en px para detectar pellizco
UMBRAL_SCROLL   = 0.03       # Umbral de movimiento para scroll
CLICK_COOLDOWN  = 0.4        # Segundos entre clics consecutivos
MARGEN          = 0.15       # Margen del área activa de la cámara (0.0–0.5)
# ─────────────────────────────────────────────────────────────

# MediaPipe Hands
mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.7
)

# Resolución de pantalla
SCREEN_W, SCREEN_H = pyautogui.size()

# ─── Estado ─────────────────────────────────────────────────
historial_x  = []
historial_y  = []
prev_scroll_y = None
ultimo_click  = 0
pausado       = False
prev_puno     = False
estado_texto  = "MOVIENDO"

def suavizar(historial, valor, max_len=SUAVIZADO):
    """Promedio móvil para suavizar el movimiento del cursor."""
    historial.append(valor)
    if len(historial) > max_len:
        historial.pop(0)
    return int(np.mean(historial))

def distancia(p1, p2, w, h):
    """Distancia euclidiana entre dos landmarks en píxeles."""
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

def dedos_levantados(landmarks, handedness):
    """
    Devuelve lista de booleanos [pulgar, índice, medio, anular, meñique].
    True = dedo levantado.
    """
    tips   = [4, 8, 12, 16, 20]   # punta de cada dedo
    pips   = [3, 7, 11, 15, 19]   # articulación media

    mano_derecha = (handedness.classification[0].label == "Right")
    levantados = []

    # Pulgar: comparar X (se invierte según mano)
    if mano_derecha:
        levantados.append(landmarks[4].x < landmarks[3].x)
    else:
        levantados.append(landmarks[4].x > landmarks[3].x)

    # Resto de dedos: comparar Y (punta más arriba que articulación)
    for tip, pip in zip(tips[1:], pips[1:]):
        levantados.append(landmarks[tip].y < landmarks[pip].y)

    return levantados

def mapear_a_pantalla(norm_x, norm_y):
    """
    Convierte coordenadas normalizadas (0–1) de la cámara
    a coordenadas de pantalla, recortando el margen activo.
    """
    # Área activa (evita los bordes de la cámara)
    x_activo = (norm_x - MARGEN) / (1 - 2 * MARGEN)
    y_activo = (norm_y - MARGEN) / (1 - 2 * MARGEN)

    # Clampar entre 0 y 1
    x_activo = max(0.0, min(1.0, x_activo))
    y_activo = max(0.0, min(1.0, y_activo))

    # Espejo horizontal (la cámara está invertida)
    x_activo = 1.0 - x_activo

    # Escalar con sensibilidad
    sx = int(x_activo * SCREEN_W * SENSIBILIDAD_X)
    sy = int(y_activo * SCREEN_H * SENSIBILIDAD_Y)

    # Clampar a resolución de pantalla
    sx = max(0, min(SCREEN_W - 1, sx))
    sy = max(0, min(SCREEN_H - 1, sy))
    return sx, sy

# ─── Loop principal ──────────────────────────────────────────
cap = cv2.VideoCapture(CAMARA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("╔══════════════════════════════════════╗")
print("║  FINGER MOUSE — Iniciado             ║")
print("║  Presiona Q para salir               ║")
print("╚══════════════════════════════════════╝")

while True:
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] No se pudo leer la cámara.")
        break

    frame = cv2.flip(frame, 1)  # Espejo para naturalidad
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    # ── Overlay base ────────────────────────────────────────
    overlay = frame.copy()

    # Zona activa (rectángulo visual)
    mx1 = int(MARGEN * w);  my1 = int(MARGEN * h)
    mx2 = int((1 - MARGEN) * w); my2 = int((1 - MARGEN) * h)
    cv2.rectangle(overlay, (mx1, my1), (mx2, my2), (80, 200, 120), 1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (80, 200, 120), 2)

    if result.multi_hand_landmarks and result.multi_handedness:
        lm         = result.multi_hand_landmarks[0]
        handedness = result.multi_handedness[0]
        landmarks  = lm.landmark

        # Dibujar esqueleto de la mano
        mp_draw.draw_landmarks(
            frame, lm,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )

        dedos = dedos_levantados(landmarks, handedness)
        # dedos = [pulgar, índice, medio, anular, meñique]
        pulgar, indice, medio, anular, menique = dedos

        indice_tip = landmarks[8]
        pulgar_tip = landmarks[4]
        medio_tip  = landmarks[12]

        dist_pellizco = distancia(indice_tip, pulgar_tip, w, h)

        now = time.time()

        # ── Detectar PUÑO (pausar/reanudar) ─────────────────
        es_puno = not any(dedos)
        if es_puno and not prev_puno:
            pausado = not pausado
            estado_texto = "⏸ PAUSADO" if pausado else "▶ REANUDADO"
        prev_puno = es_puno

        if not pausado and not es_puno:

            # ── MODO SCROLL: índice + medio levantados ───────
            if indice and medio and not anular and not menique:
                estado_texto = "SCROLL"
                cx = suavizar(historial_x, indice_tip.x)
                cy = indice_tip.y

                if prev_scroll_y is not None:
                    delta = cy - prev_scroll_y
                    if abs(delta) > UMBRAL_SCROLL:
                        scroll_amt = int(-delta * 20)
                        pyautogui.scroll(scroll_amt)

                prev_scroll_y = cy

                # Punto visual en índice
                px = int(indice_tip.x * w)
                py = int(indice_tip.y * h)
                cv2.circle(frame, (px, py), 12, (255, 200, 0), -1)
                cv2.putText(frame, "SCROLL", (px - 30, py - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)

            # ── CLIC DERECHO: 5 dedos abiertos ───────────────
            elif all(dedos):
                estado_texto = "CLIC DERECHO"
                if now - ultimo_click > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    ultimo_click = now
                cv2.putText(frame, "RIGHT CLICK", (10, h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

            # ── CLIC IZQUIERDO: pellizco índice+pulgar ────────
            elif indice and not medio and dist_pellizco < UMBRAL_CLICK:
                estado_texto = "CLIC"
                if now - ultimo_click > CLICK_COOLDOWN:
                    pyautogui.click()
                    ultimo_click = now

                # Dibujar línea de pellizco
                p1 = (int(indice_tip.x * w), int(indice_tip.y * h))
                p2 = (int(pulgar_tip.x * w), int(pulgar_tip.y * h))
                cv2.line(frame, p1, p2, (0, 255, 255), 3)
                cv2.circle(frame, p1, 10, (0, 255, 255), -1)
                cv2.circle(frame, p2, 10, (0, 255, 255), -1)

            # ── MOVER CURSOR: solo índice levantado ───────────
            elif indice and not medio:
                estado_texto = "MOVIENDO"
                prev_scroll_y = None

                sx, sy = mapear_a_pantalla(indice_tip.x, indice_tip.y)
                sx = suavizar(historial_x, sx)
                sy = suavizar(historial_y, sy)

                pyautogui.moveTo(sx, sy)

                # Punto visual
                px = int(indice_tip.x * w)
                py = int(indice_tip.y * h)
                cv2.circle(frame, (px, py), 15, (0, 255, 0), -1)
                cv2.circle(frame, (px, py), 18, (255, 255, 255), 2)

    # ── HUD ─────────────────────────────────────────────────
    color_estado = {
        "MOVIENDO":     (0, 255, 100),
        "CLIC":         (0, 255, 255),
        "SCROLL":       (255, 200, 0),
        "CLIC DERECHO": (0, 100, 255),
        "⏸ PAUSADO":   (100, 100, 100),
        "▶ REANUDADO":  (0, 200, 255),
    }.get(estado_texto, (255, 255, 255))

    # Panel HUD
    cv2.rectangle(frame, (0, 0), (300, 30), (20, 20, 20), -1)
    cv2.putText(frame, f"MODO: {estado_texto}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_estado, 2)

    cv2.rectangle(frame, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "Q: Salir | Puno: Pausar | Pellizco: Click | 2 dedos: Scroll | 5 dedos: Click Der",
                (6, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    cv2.imshow("Finger Mouse — Alex", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Finger Mouse cerrado.")