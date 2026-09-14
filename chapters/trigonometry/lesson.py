import numpy as np

angle = 45.0
length_u = 1.5
length_v = 1.0

def update(angle, length_u, length_v):
    if not np.isfinite([angle, length_u, length_v]).all() or min(length_u, length_v) < 0:
        raise ValueError("Use a finite angle and nonnegative vector lengths")
    theta = np.deg2rad(np.float32(angle))
    cosine = np.cos(theta)
    sine = np.sin(theta)
    tangent = np.tan(theta) if abs(cosine) > 1e-6 else np.float32(np.nan)
    u = np.array([length_u, 0], dtype=np.float32)
    v = np.float32(length_v) * np.array([cosine, sine], dtype=np.float32)
    projection = np.array([v[0], 0], dtype=np.float32)
    dot = u @ v
    cross_z = u[0] * v[1] - u[1] * v[0]
    similarity = cosine if length_u > 0 and length_v > 0 else np.float32(np.nan)

    print(f"Angle = {angle:g} degrees = {theta:.4f} radians")
    print(f"sin = {sine:.4f}, cos = {cosine:.4f}, tan = {tangent:.4f}")
    print(f"u dot v = {dot:.4f}")
    print(f"Cosine similarity = {similarity:.4f}")
    print(f"(u cross v)_z = {cross_z:.4f}; area = {abs(cross_z):.4f}")

    return {
        "INPUTS": {"u": u, "v": v},
        "OUTPUTS": {
            "Angle": np.array([angle, theta], dtype=np.float32),
            "Unit point": np.array([cosine, sine], dtype=np.float32),
            "Tangent": np.float32(tangent) if np.isfinite(tangent) else np.float32(0),
            "Tangent defined": int(np.isfinite(tangent)),
            "Projection": projection,
            "Dot product": dot,
            "Cross z": cross_z,
            "Cosine similarity": np.array([similarity] if np.isfinite(similarity) else [], dtype=np.float32),
        },
    }


PARAMETERS = {"angle": angle, "length_u": length_u, "length_v": length_v}
globals().update(update(**PARAMETERS))

degrees = np.linspace(-360, 360, 1441, dtype=np.float32)
radians = np.deg2rad(degrees)
tan_curve = np.where(np.abs(np.cos(radians)) > 1e-6, np.tan(radians), np.nan)

PLOTS = {
    "Sine and cosine": {
        "sin": np.column_stack((degrees, np.sin(radians))),
        "cos": np.column_stack((degrees, np.cos(radians))),
    },
    "Tangent": {"tan": np.column_stack((degrees, tan_curve))},
}