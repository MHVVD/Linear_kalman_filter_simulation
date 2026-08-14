import cv2
import numpy as np

# --- 1. CANVAS & MOUSE SETUP ---
# Create a black canvas
frame = np.zeros((800, 800, 3), np.uint8)
true_mouse_pos = np.array([400, 400], dtype=np.float32)

# Mouse callback to capture "Ground Truth" (Where the mouse ACTUALLY is)
def get_mouse_pos(event, x, y, flags, param):
    global true_mouse_pos
    if event == cv2.EVENT_MOUSEMOVE:
        true_mouse_pos = np.array([x, y], dtype=np.float32)

cv2.namedWindow("Kalman Filter: Noisy Mouse")
cv2.setMouseCallback("Kalman Filter: Noisy Mouse", get_mouse_pos)


# --- 2. THE KALMAN FILTER MATRICES ---
# This matches your handwritten notes exactly!
dt = 1.0

# Initial State [X_pos, Y_pos, X_velocity, Y_velocity]
x_k = np.array([[400], [400], [0], [0]], dtype=np.float32) 

# State Transition Matrix (F) - Physics model assuming constant velocity
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1,  0],
              [0, 0, 0,  1]], dtype=np.float32)

# Observation Matrix (H) - We only "sense" position, not velocity
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]], dtype=np.float32)

# Initial Covariance (P) - High initial uncertainty
P_k = np.eye(4, dtype=np.float32) * 1000

# Process Noise (Q) - Small random bumps in our physics model
Q = np.eye(4, dtype=np.float32) * 0.1

# Measurement Noise (R) - HUGE sensor inaccuracy (This simulates our terrible GPS)
R = np.eye(2, dtype=np.float32) * 800

I = np.eye(4, dtype=np.float32)
last_est_pos = None

print("Move your mouse around the window! Press 'q' to quit.")

# --- 3. THE MAIN LOOP ---
while True:
    # --- SIMULATING THE SENSOR ---
    # Add heavy random noise to true mouse position to simulate a bad GPS (z_k)
    noise = np.random.randn(2) * 40  
    z_k = (true_mouse_pos + noise).reshape(2, 1)

    
    # --- THE MATH (From your notebook) ---
    
    # STEP 1: PREDICT
    x_k = F @ x_k 
    P_k = F @ P_k @ F.T + Q

    # STEP 2: UPDATE (Measurement & Gain)
    y_k = z_k - (H @ x_k)                      # Innovation
    S_k = H @ P_k @ H.T + R                    # Innovation Covariance
    K = P_k @ H.T @ np.linalg.inv(S_k)         # Kalman Gain
    
    x_k = x_k + (K @ y_k)                      # Final State Update
    P_k = (I - K @ H) @ P_k                    # Final Covariance Update


    # --- VISUALS ---
    frame = frame * 0.92  # Gives a nice fading "trail" effect
    frame = np.uint8(frame)

    est_x, est_y = int(x_k[0, 0]), int(x_k[1, 0])
    meas_x, meas_y = int(z_k[0, 0]), int(z_k[1, 0])
    true_x, true_y = int(true_mouse_pos[0]), int(true_mouse_pos[1])

    # 1. Draw True Position (Faint Green Dot)
    cv2.circle(frame, (true_x, true_y), 5, (0, 255, 0), -1)
    
    # 2. Draw Noisy Sensor Reading (Red Cross)
    cv2.drawMarker(frame, (meas_x, meas_y), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)

    # 3. Draw Kalman Filter Estimate (Thick Blue Line)
    if last_est_pos is not None:
        cv2.line(frame, last_est_pos, (est_x, est_y), (255, 0, 0), 4)
    last_est_pos = (est_x, est_y)

    cv2.imshow("Kalman Filter: Noisy Mouse", frame)

    # 30 FPS, press 'q' to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
