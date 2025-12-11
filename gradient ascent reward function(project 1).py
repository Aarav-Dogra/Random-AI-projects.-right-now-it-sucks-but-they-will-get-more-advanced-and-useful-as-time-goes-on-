#K,H&P are variables that are meant to be optimized, I gave the 'AI' a simple/typical reward function, a guassian distribution
import random as r
e = 2.71828182846
k=0.0
h=0.0
p=0.0
Δ = 0.05

# capital letters show what each property was at a certain t, like K[t]
K = [0.0]
H = [0.0]
P = [0.0]
R = [0.0]
t = 0

Reward_function = 0.0
w = 16 #how harsh the reward function is (deviation size in gaussian), unexpectedly higher w can lead to smaller step sizes, but this makes sense coz it tells u whether ur going in the right direction or not alot more "explicityly", the sensitivity is higher, so u have no choice but to figure out the best path, coz u explore more(randomness) possible gradients at the start which leads to a higher probability of a really good gradient(which we keep in this code), and unsuprisingly makes it more variable
while Reward_function < 0.99:
    # current reward from environment (agent only sees this scalar)
    Reward_function = e**(-w*(k-0.8)**2) * e**(-w*(h-0.15)**2) * e**(-w*(p-0.05)**2)
    t += 1

    # track current state
    K.append(k)
    H.append(h)
    P.append(p)
    R.append(Reward_function)

    # --- UPDATED SECTION (no gradient; RL-style acceptance) ---
    # propose a small exploratory move
    noise_scale = max(0.0, 1.0 - Reward_function)  # less exploration near optimum
    dk = noise_scale * r.uniform(-Δ, Δ)
    dh = noise_scale * r.uniform(-Δ, Δ)
    dp = noise_scale * r.uniform(-Δ, Δ)

    # candidate step
    k_candidate = min(1.0, max(0.0, k + dk))
    h_candidate = min(1.0, max(0.0, h + dh))
    p_candidate = min(1.0, max(0.0, p + dp))

    # get the new reward (environment feedback)
    new_reward = e**(-w*(k_candidate-0.8)**2) * e**(-w*(h_candidate-0.15)**2) * e**(-w*(p_candidate-0.05)**2)

    # accept if improves; else reject (or very rarely accept to escape plateaus)
    if new_reward >= Reward_function or r.random() < 0.05:  # small chance to explore
        k, h, p = k_candidate, h_candidate, p_candidate
        # optional: small Δ decay as we get better
        Δ = max(0.005, Δ * (0.999 if new_reward > Reward_function else 1.0))
    # --- END UPDATED SECTION ---

print(f"Steps: {t}")
print(f"Final values -> Knowledge: {k:.3f}, Health: {h:.3f}, Purpose: {p:.3f}")
print(f"Final Reward: {Reward_function:.4f}")
