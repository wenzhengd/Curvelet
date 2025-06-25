# Curvelet Control for Qubit Decoherence Mitigation

**A geometric-wavelet hybrid framework for smooth and robust quantum control**

---

![Concept Diagram](https://i.imgur.com/nL9P1ot.png)

---

## 🧠 Background & Motivation

In quantum systems, **dephasing decoherence** remains a fundamental challenge that undermines qubit coherence over time. Traditional solutions like **dynamical decoupling** rely on fast, discrete control pulses, while **numerical optimal control** techniques often sacrifice physical interpretability for performance.

This project proposes a novel alternative: a **geometrically guided, wavelet-augmented framework** for designing control protocols. We aim to explore realistic, **smooth control curves** (dubbed **tantrices**) using **wavelet-based signal analysis**—an approach that is both physically meaningful and analytically tractable.

---

## 🔁 Curvelet Framework: Geometry Meets Wavelets

At the heart of this method lies the **Curvelet**, a unification of:

- **Tantrix** – A smooth geometric curve representing a physically realizable control path.
- **Wavelets** – Analytic-numerical basis functions that allow localized, multi-scale control refinement.

This hybrid offers both **interpretability** and **robustness**, allowing the control path to evolve through a self-correcting feedback loop guided by coherence criteria.

---

## 🔧 Iterative Control Optimization Pipeline

The control optimization proceeds as an iterative feedback loop between the **real tantrix space** and a **wavelet-based control manifold** \( \mathcal{M}_\mathcal{F} \). Here's a step-by-step view, as illustrated in the figure:

1. **Start with a real tantrix** \( r(t)^{(i)} \):  
   An initial control ansatz, constrained to be smooth and realistic.

2. **Wavelet Expansion**:  
   This control is expanded in the wavelet basis, mapping it onto the current manifold \( \mathcal{M}_\mathcal{F}^{(i)} \). This process reveals how well it resists decoherence by checking for **coherence leakage**.

3. **Leakage Check & Manifold Surgery** 🤖:  
   If significant decoherence is detected, a feedback mechanism:
   - Identifies leakage pathways.
   - Expands the manifold to \( \mathcal{M}_\mathcal{F}^{(i+1)} \), accommodating more expressive wavelet functions to better approximate the ideal control.

4. **Map & Normalize to Pseudo-Tantrix** \( \tilde{r}(t)^{(i)} \):  
   The improved wavelet solution is mapped back to a physically meaningful tantrix, through a pseudo tantrix transitionally , preserving smoothness.

5. **Re-injection for Iteration**:  
   The new tantrix is fed back for another cycle of wavelet projection and refinement.

🔁 **This iterative loop continues until the decoherence is sufficiently suppressed.**  
🧠 **Convergence is not required**; useful, robust control paths can emerge in finite iterations.

---

## 📌 Highlights

- **No bang-bang pulses** – Only smooth control trajectories compatible with experimental hardware.
- **No blind numerics** – Analytical understanding through wavelet theory and geometric intuition.
- **Adaptivity** – Feedback-driven control space expansion allows flexible, targeted improvements.

---

## 📂 Repository Contents

- `curvelet_optimizer.py` – Main iterative control search implementation.
- `wavelet_utils.py` – Tools for constructing and expanding wavelet manifolds.
- `tantrix_geometry.py` – Geometric utilities for smooth curve parametrization.
- `examples/` – Sample cases for various qubit dephasing environments.

---

## 📖 Suggested Reading

- Wavelet transforms in quantum control and signal processing  
- Geometric control theory in open quantum systems  
- Coherence projection techniques in decoherence modeling

---

## 📬 Contact

For questions or collaboration inquiries, feel free to reach out via [wenzheng.dong.quantum@gmail.com] or connect on [LinkedIn](https://www.linkedin.com/in/wenzheng-dong).

