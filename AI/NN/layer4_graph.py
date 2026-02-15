import matplotlib.pyplot as plt

# 填入你提供的數據
data = [
    (0.00, 2.053768), (0.10, 2.031401), (0.20, 1.968706), (0.30, 1.872703),
    (0.40, 1.755524), (0.50, 1.626535), (0.60, 1.495602), (0.70, 1.369077),
    (0.80, 1.247147), (0.90, 1.128201), (1.00, 1.012643), (1.10, 0.903180),
    (1.20, 0.803039), (1.30, 0.714251), (1.40, 0.637141), (1.50, 0.570793),
    (1.60, 0.513728), (1.70, 0.464388), (1.80, 0.421367), (1.90, 0.383482),
    (2.00, 0.349772)
]

x_val, y_val = zip(*data)

plt.figure(figsize=(10, 6))
plt.plot(x_val, y_val, marker='o', linestyle='-', color='b', label='NN Approximation f(x)')
plt.axhline(1, color='red', linestyle='--', alpha=0.3)
plt.axvline(1, color='red', linestyle='--', alpha=0.3)
plt.title("Iterative Function Solution: f(f(x)) = (2x^2 + 1)/3")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.show()