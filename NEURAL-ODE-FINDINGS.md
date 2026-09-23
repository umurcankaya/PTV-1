# ProteinTalks — neural ODE findings

Working notes on the neural ODE in Sun et al., *An operational perturbation proteomics-based
virtual cell model*, Nature 2026 (doi:10.1038/s41586-026-11001-9), against the code at
`ProteinTalks/` and the released `best_checkpoint.pth`.

Scope: the ODE component only. The perturbation dataset (16,311 proteomes, 38M measurements),
the siRNA/organoid validation and the benchmark comparisons are out of scope here and are not
challenged by anything below.

**Confidence legend**

| tag         | meaning                                                                        |
| ----------- | ------------------------------------------------------------------------------ |
| `[READ]`  | verified by reading released source; anyone can check in minutes               |
| `[CKPT]`  | measured from`best_checkpoint.pth`; needs a clean re-run with `torch.load` |
| `[INFER]` | derived from architecture or from Methods prose;**not measured**         |

---

## 0. The central discrepancy

Methods eq. (4), p. 13:

```
z₀     = C₁
dz(t)/dt = f_θ(z(t), t, D)
z(t_k) = ODESolve(f_θ, z₀, [0, t_k]; RK4),   t_k ∈ {6, 24, 48}
```

`model.py:59-65` and `model.py:126`:

```python
func = nn.Sequential(
    FullyConnectedLayer(hidden_feats, hidden_feats, nn.Softplus(), dropout),
    FullyConnectedLayer(hidden_feats, hidden_feats, None,          dropout),
)
self.neuralDE = NeuralODE(func, solver='rk4')
...
emb_ode = self.neuralDE(emb_cnn, torch.linspace(0, self.time_tick_num-1, self.time_tick_num))
```

`nn.Sequential.forward()` accepts exactly one argument, so neither `t` nor `D` can reach the
vector field. `linspace(0, 3, 4)` is `[0., 1., 2., 3.]`. The equation and the implementation
describe different objects.

---

## F1 — The field is autonomous `[READ]`

$\dot z = f_\theta(z)$, not $f_\theta(z,t,D)$.

Two consequences that fail independently and should be argued separately:

**F1a — no `t`.** The flow depends only on elapsed integration time.

**F1b — no `D`.** The paper claims a *controlled* system $\dot z = f(z,u)$; the code delivers
$\dot z = f(z)$ with $z_0 = g(u)$. The drug acts as an impulse at $t=0$ and is then absent from
the dynamics — it cannot represent sustained exposure. `D` is also only a binary target mask:
the 935-dim fingerprints never enter Module 1, so **two drugs with identical target sets produce
byte-identical proteome trajectories**. No dose, no structure.

---

## F2 — Uniform index grid, not physical time `[READ]`

`linspace(0, 3, 4)` with `time_tick_num = 4` (`model.py:71`). Physical hours never enter the
model as numbers — they are only CSV row selectors in `dataset.py:48`. The binding is positional,
established solely by `F.mse_loss(outputs, y)` at `trainer.py:124`.

With F1a this means the same time-1 map Φ is applied three times:

```
z₁ = Φ(z₀)  ──►  P(6h)     real gap  6 h
z₂ = Φ(z₁)  ──►  P(24h)    real gap 18 h
z₃ = Φ(z₂)  ──►  P(48h)    real gap 24 h
```

### The time-warp defence, and why it fails

Steelman: a uniform grid is just a reparameterisation $\tau = g(t)$, equivalent to
$\dot z = g'(t) f_\theta(z)$ — a legitimate non-autonomous model with separable rate.
This is sound, and it means "the model equates 18 h with 6 h" is *not* the right criticism.

It fails on three points:

1. **Unidentifiable at three taps.** With a free-form Φ, every monotone $g$ gives the identical
   model class. The warp explains nothing and predicts nothing.
2. **Contradictory across datasets `[INFER]`.** PTDS (0,6,24,48 h) makes 0→6 h one τ-unit and
   24→48 h one τ-unit. mtPTDS (0,2,…,60 h, 11 points) makes 0→6 h **three** units and 24→48 h
   **two**. Same biology, overlapping cell lines (HCC1806, HCC1143, MDA-MB-453). A real warp
   gives one answer.
3. **It tracks the protocol.** $g'(t) \propto 1/\Delta t_\text{sampling}$ always. In mtPTDS the
   implied $g$ has a sixfold rate discontinuity at exactly 12 h — where sampling switches from
   2-hourly to 12-hourly.

Note: the mtPTDS variant is not in the released code (`time_tick_num` is hardcoded to 4,
`config.py:28` restricts `--time_stamp_predict_drug` to `["6_24_48"]`). `linspace(0,10,11)` is
inferred from Methods p. 14.

Also: paper p. 5 reports that going from 4 to 11 time points *decreased* AUPRC (P = 0.02).

---

## F3 — Is a fixed-step RK4 flow an ODE? `[READ]` + `[INFER]`

One RK4 step is $z' = z + G(z)$ where $G(z) = \tfrac{h}{6}(k_1+2k_2+2k_3+k_4)$ — a residual
block whose branch is a four-stage weight-tied composition of $f$. **Not** the literal
Euler/ResNet equivalence. Three steps ≈ 12 sequential $f$-evals, ~24 linear layers, period-2
tying. Genuinely more than a ResNet block.

But it is not more *ODE*:

- Solver order is accuracy relative to a *given* $f$. Here $f$ is learned and only three taps
  are supervised — there is no reference trajectory.
- No small parameter at $h=1$: see F4, $\|f\| \gtrsim 2.5$, so one step moves the 64-d state by
  a distance comparable to its own norm.
- Nothing treats it as an integrator: `--tol` (`config.py:45`) is never read, the solver is
  non-adaptive, no step-size study.

**Open: does torchdyn 1.0.4 take 3 steps of h=1 on this t_span?** `[INFER]` — asserted, never
instrumented. The index-grid argument survives either way; "3 RK4 steps" might not.

**Non-problem:** torchdyn defaults to `sensitivity='autograd'` (backprop through solver, not
adjoint). At depth 3 that is the correct choice, and it sidesteps the adjoint accuracy problem
Gholami et al. document.

---

## F4 — The field cannot vanish: no equilibria `[CKPT]`

`FullyConnectedLayer` terminates in `nn.LayerNorm`, and the vector field is two of them stacked,
so the last operation applied to the *derivative* is normalisation:

$$
f(z) = \gamma \odot \hat z + \beta, \qquad \|\hat z\| = \sqrt{64} = 8
$$

From `neuralDE.vf.vf.vf.1.norm.*` in the checkpoint: $\min|\gamma| = 0.4065$,
$\max|\gamma| = 0.9575$, $\|\beta\| = 0.7207$, giving

$$
\|f(z)\| \in [2.53,\ 8.38] \quad \forall z
$$

Strictly positive everywhere ⇒ **$f(z)=0$ has no solution ⇒ no fixed points**. No steady state,
no relaxation, no attractor involving deceleration. The only admissible ω-limit sets are limit
cycles or wandering trajectories.

Nuance: this forbids stopping, not growth. Direction is unconstrained, so the latent may curve or
orbit. And because the decoder is linear, a *predicted protein abundance* can still plateau by
moving the latent within the decoder's null space. What is unrepresentable is genuine
equilibration — and extrapolation past the training horizon must drift.

Almost certainly unintended: an artifact of reusing a layer class that always appends LayerNorm.

*(Earlier note said [5.8, 7.2]. That used $\|\gamma\| \pm \|\beta\|$, which is not a valid bound
for an elementwise product. [2.53, 8.38] is correct; conclusion unchanged.)*

---

## F5 — 5,585 independent copies of a 64-d ODE `[READ]`

Every conv is `kernel_size=1` (`model.py:51,53`), and `linear_input` takes `in_feats=2`. So the
state is 5,585 independent 64-d systems sharing one $f$, coupled only through
`GroupNorm(1, 64)` (`model.py:52`) — which with `num_groups=1` normalises over channels *and*
all protein positions, i.e. couples via exactly two sample-level scalars $(\mu, \sigma)$.

Consequences:

- $\hat y_p(t) = \Phi_t(x_p, d_p)$ — each protein's entire predicted trajectory is a
  deterministic function of two scalars. The forecasting branch is a map $\mathbb{R}^2 \to \mathbb{R}^3$:
  two univariate transfer curves, one for targets and one for everything else.
- The off-diagonal Jacobian $\partial \hat P_i(t)/\partial x_j$ has **rank ≤ 2** `[INFER]`.
- Protein *p*'s response cannot depend on protein *q*'s baseline. "TYMS rises because this cell
  has high thymidylate flux" is unrepresentable.
- All conditions flow under one field, so trajectories cannot cross (Dupont et al.). If two
  conditions coincide in latent space at any point, their futures are identical.

**Balance.** This is a deliberate, *stated* design choice — Methods p. 13 says the point-wise
convolution "ties each protein's embedding to its own features instead of mixing neighbouring
proteins," and keeping parameters independent of protein count genuinely helps transfer.
Weight-shared pointwise functions are standard (DeepSets, 1×1 convs, per-token MLPs). It is not
a bug. The problem is the mismatch with the narrative: the model is called ProteinTalks, Fig. 1
labels this module "Dynamic protein network," and the abstract sells "dynamical latent
representations." In the dynamics, proteins do not interact. If interaction were wanted, the
established tool is a graph neural ODE (Poli et al.), where the field is a function of the
graph's connectivity.

---

## F6 — Dropout inside the vector field `[READ]`

`FullyConnectedLayer` includes `nn.Dropout`, and the vector field is two of them. RK4 evaluates
$f$ four times per step, and each call draws an **independent** mask — so the four stages
evaluate four *different* functions. RK4's order conditions assume one fixed $f$; they are void.
Because $f$ is nonlinear, the expected step is not even the RK4 step of the mean field, so the
scheme is consistent with no ODE at all. Train mode solves a stochastic non-ODE; eval mode solves
a deterministic RK4 — different systems.

Not standard practice. Liu et al. state the position directly: dropout and Gaussian noise "are
missing in current Neural ODE networks because their deterministic nature prevents the
incorporation of stochastic regularization techniques" — the principled route is a neural SDE
with an SDE solver, or noise outside the ODE block.

Mitigating: the sweep (Methods p. 14) covered dropout ∈ {0, 0.1, 0.2} and the best config used 0.
Aggravating: `ppODE.__init__` defaults to `dropout=0.1`, so anyone reusing the class directly
gets a broken integrator.

---

## F7 — The latent is never anchored at t = 0 `[READ]`

`model.py:128`:

```python
emb_ode = emb_ode[1][1:]      # drops the t=0 state before decoding
```

The $t=0$ state is discarded and never passed through the decoder, and there is no reconstruction
term anywhere in `trainer.py`. So nothing forces $\text{dec}(z_0) \approx P_{0h}$.

Why it matters: the encoder may map the baseline proteome anywhere in $\mathbb{R}^{64}$, however
distorted, provided that after 1, 2, 3 applications of Φ the decoder emits the right numbers. The
curve therefore touches proteome space at three sampled instants and nowhere else — including not
at its own origin. Calling it "the proteome's trajectory" presumes an anchor that was never
imposed.

Downstream: Fig. 4a's dynamical SHAP computes $\partial \hat P(t)/\partial P(0)$ and reads it as
"how the baseline state shapes the response," which presumes the trajectory starts at the baseline
state. Extended Data Fig. 4a's "gradual, ordered transition across the ten post-treatment states"
is ordering *latents* — and a continuous flow orders its own latents by construction.

Standard latent-ODE practice (Chen et al. 2018; Rubanova et al. 2019) reconstructs at all observed
times including $t=0$. This is a one-line omission.

---

## F8 — The dynamics term is ~1% of the objective `[READ]` + `[CKPT]`

`trainer.py:131`: `loss = 0.2 * mse + 0.8 * bce`, with MSE taken over min–max-normalised $[0,1]$
targets, so the weighted proteome term lands around $10^{-3}$. The checkpoint's
`scheduler_state_dict` records best val loss **0.22423 ≈ 0.8 × BCE**. Model selection is ~99%
driven by the classifier.

Related (`dataset.py:85-97`): targets are log-transformed then min–max normalised **per sample and
per timepoint independently**, removing proteome-wide drift. With biological-replicate Pearson at
0.92–0.96 (their Extended Data Fig. 1B), $\hat y = x$ would score near ceiling. No identity
baseline is reported.

---

## F9 — `drugsens_conv1` is frozen at initialisation `[CKPT]`

Not strictly an ODE finding, recorded because it bounds what the ODE branch can possibly
contribute downstream.

`Conv1d(5585, 32, kernel_size=4)` — 714,912 params, **90.5% of the model's 790,306** — is the sole
route from the predicted trajectory to the phenotype head. In `best_checkpoint.pth` it has
**no `exp_avg`, no `exp_avg_sq`, no `step`**, while all 26 other parameters record `step = 19771`.
PyTorch's AdamW does `if p.grad is None: continue` before state init, so it never received a
gradient. The only other parameter in that state is `convdrug1`, which `forward()` never calls.

Its weights are exact Kaiming-uniform init: $\max|w| = 1/\sqrt{22340} = 0.0066905$ to six digits,
excess kurtosis −1.1984 (uniform = −1.2), χ² = 51.2 on 49 dof, zero mass outside the init support.
For contrast, `drugs_conv2` (fingerprints) has 38% of weights outside its init support and
kurtosis +58.

Mechanism: `set_time_stamp_predict_drug` (`model.py:79-92`) rebinds `self.drugsens_conv1` to a
fresh `nn.Conv1d`. Called after the optimiser is built, the old tensors stay in the optimiser
(no grad) and the new ones are never registered (frozen). Corroborating: the saved `weight_decay`
is 0.01 (AdamW's default), not the 1e-4 `main.py` passes.

**Caveat, load-bearing:** the released `main.py` never calls that method, so retraining from the
public repo *would* train this layer. This is a defect in whatever script produced the checkpoint,
not in the published architecture — and we cannot attribute this checkpoint to any published
figure.

Consistent behavioural signature (paper p. 5): leave-one-cell-line-out AUROC 0.95, random split
0.92, **leave-one-drug-out 0.64**. (LODO uses single agents only, so not a clean swap.)

---

## Open tests, ranked

| #  | test                                                                                                                        | settles                                                                           |
| -- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 1  | Refine the grid: re-integrate frozen$f_\theta$ on `linspace(0,3,31)`, read indices 10/20/30, compare to $h=1$ outputs | whether a well-defined vector field was learned at all (F3)                       |
| 2  | Spectrum of$\partial f/\partial z$ (64×64) vs RK4's real-axis stability limit $\vert\lambda\vert h \lesssim 2.78$      | sharper than#1: a stepper outside the stability region cannot be a discretisation |
| 3  | Off-diagonal Jacobian$\partial\hat y_i/\partial x_j$, strip diagonal, SVD                                                 | rank ≤ 2 prediction (F5); decides Fig. 4a                                        |
| 4  | Integrate to τ = 10, 50                                                                                                    | F4: predictions must keep drifting, never settle                                  |
| 5  | Re-read checkpoint with`torch.load`                                                                                       | confirms F9 with the standard loader                                              |
| 6  | Retrain with`drugsens_conv1` registered in the optimiser                                                                  | whether published numbers are affected                                            |
| 7  | Seed swap, top-100 SHAP overlap                                                                                             | whether Fig. 4 / survival panel survive                                           |
| 8  | Feed*measured* 6/24/48 h proteomes to the phenotype head                                                                  | whether Module 2 can use proteomics at all                                        |
| 9  | Identity baseline$\hat y = x$ for MSE and per-sample Pearson                                                              | F8                                                                                |
| 10 | Zero the fingerprints                                                                                                       | expected to be the ablation that actually hurts                                   |

---

## References

- Chen, Rubanova, Bettencourt & Duvenaud. *Neural Ordinary Differential Equations.* NeurIPS 2018.
  arXiv:1806.07366. — baseline formulation; also ref. 22 in the paper.
- Dupont, Doucet & Teh. *Augmented Neural ODEs.* NeurIPS 2019. arXiv:1904.01681. — ODE flows are
  homeomorphisms, trajectories cannot cross, hence there exist functions neural ODEs cannot
  represent. Relevant to **F5**.
- Gholami, Keutzer & Biros. *ANODE: Unconditionally Accurate Memory-Efficient Gradients for Neural
  ODEs.* IJCAI 2019. arXiv:1902.10298. — adjoint gradients can be inaccurate/unstable; proposes
  checkpointing. Note this *supports* ProteinTalks' use of backprop-through-solver (**F3**,
  non-problem); it is not about protein coupling.
- Liu et al. *Neural SDE: Stabilizing Neural ODE Networks with Stochastic Noise.* arXiv:1906.02355.
  — dropout/Gaussian noise are absent from neural ODEs because determinism precludes them; the
  principled route is a neural SDE. Relevant to **F6**.
- Oganesyan, Volokhova & Vetrov. *Stochasticity in Neural ODEs: An Empirical Study.*
  arXiv:2002.09779. — empirical treatment of stochastic regularisation in neural ODEs. **F6**.
- Kidger. *On Neural Differential Equations.* DPhil thesis, arXiv:2202.02435. — canonical
  reference; noise should be handled as an SDE with a fixed (not learned) diffusion. **F6**.
- Poli et al. *Graph Neural Ordinary Differential Equations.* arXiv:1911.07532. — the established
  construction when you want the field to depend on interactions between units. **F5**.
- Ott, Katiyar, Hennig & Tiemann. *ResNet After All? Neural ODEs and Their Numerical Solution.*
  ICLR 2021. — neural ODEs trained with coarse fixed-step solvers learn discretisation-dependent
  fields; proposes the refinement check. **F3 / open test #1.** *(Title and author list recalled
  from memory — verify before citing.)*

---

## What we are NOT claiming

- That any published number is wrong. F9 concerns one checkpoint we cannot tie to any figure.
- That the dataset or the wet-lab validation is affected. Neither depends on the ODE.
- That the architecture is unusable. F5 and F3 are defensible engineering; the objection is to the
  gap between them and the paper's description.
- Misconduct. The Methods text is in places *more* honest than the figure captions (p. 13 states
  the point-wise convolution does not mix proteins), and the authors reported the LODO collapse
  rather than burying it. This reads as drift between writing and implementation.

The strongest, most rebuttal-proof material is F1, F2, F5, F6, F7 — all `[READ]`, all checkable
against the published equations in minutes.
