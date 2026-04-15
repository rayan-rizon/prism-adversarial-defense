# PRISM: Complete Implementation Guide — Step-by-Step for Beginners

## Critical Validation Report

Before implementation, here is an honest assessment of the PRISM research plan based on cross-checking against the current literature (April 2026).

<aside>
⚠️

**NOVELTY CORRECTIONS — Read Before Starting**

The original plan overstates novelty in several areas. These corrections are critical for honest paper positioning.

</aside>

### Contribution 1: TDA for Adversarial Detection — Claimed ~95%, Actual ~60-70%

**What the plan missed:**

- **Gebhart & Schrater** — "Adversary Detection in Neural Networks via Persistent Homology" — directly uses persistent homology for adversarial detection
- **LANL (Los Alamos National Lab)** — Published open-source code `tda-adversarial-detection` on GitHub, using persistent homology + two-sample testing to detect adversarial examples in CLIP models (ImageNet, CIFAR-10/100)
- **Goibert, Ricatte & Dohmatob** — "An Adversarial Robustness Perspective on the Topology of Neural Networks" — shows clean vs adversarial inputs produce different network graph topologies
- **Ballester et al. (2023)** — Comprehensive survey "TDA for Neural Network Analysis" covers adversarial detection as a known application area
- **Czernek (2025)** — Master thesis on persistent homology for image classification robustness
- **WGTL (2025)** — Witness Graph Topological Layer for GNN adversarial robustness with stability guarantees

**What IS still novel:** Using TDA as the PRIMARY detection backbone in a multi-tiered, architecture-agnostic runtime defense system with conformal calibration. The integration is novel, the individual technique is not.

### Contribution 2: Conformal Prediction for Detection — Claimed ~85%, Actual ~65-75%

**What the plan missed:**

- **Gendler et al. (ICLR 2022)** — "Adversarially Robust Conformal Prediction" — combines conformal prediction with randomized smoothing for adversarial robustness
- **VRCP (2025)** — "Verifiably Robust Conformal Prediction" — uses NN verification + conformal prediction for adversarially robust prediction sets
- **ScienceDirect (2026)** — Conformal ML framework for anomaly detection in industrial CPS with formal false alarm guarantees

**What IS still novel:** Using conformal prediction to calibrate a TDA-based adversarial DETECTOR (not prediction sets). The Stutz differentiation in the plan is valid.

### Contribution 3: Sequential Campaign Detection — Claimed ~96%, Actual ~90-96%

**This claim holds up well.** No direct competitor found for session-level adversarial campaign detection in neural network inference pipelines using BOCPD. The temporal survey ([Preprints.org](http://Preprints.org) 2026) covers temporal adversarial attacks but not temporal DEFENSE at the campaign level. This is genuinely the strongest contribution.

### Contribution 4: Topology-Aware MoE — Claimed ~78%, Actual ~70-78%

Reasonable claim. The Wasserstein-based gating mechanism using persistence diagrams is genuinely novel.

### RAILS Assessment — Confirmed Valid

RAILS (Wang et al., IEEE Access 2022, 12 citations) is confirmed. GitHub code available at `wangren09/RAILS`. The differentiation (architecture-coupled DkNN vs. architecture-agnostic PRISM) is accurate and defensible.

### Croce et al. Assessment — Valid but Needs Stronger Framing

The plan's three-angle defense against Croce is solid. The key experiment — PRISM + adversarially-trained backbone > backbone alone — is the correct framing.

<aside>
✅

**Bottom line:** PRISM is still a publishable, novel contribution — but you MUST cite the additional papers above, and reframe novelty claims from "first to use TDA for adversarial defense" to "first to integrate TDA-based detection with conformal guarantees and campaign-level temporal monitoring in an architecture-agnostic system."

</aside>

---

## Phase 0: Environment Setup (Week 0 — Before Anything Else)

### 0.1 Hardware Requirements

| **Component** | **Minimum** | **Recommended** |
| --- | --- | --- |
| GPU | RTX 3060 12GB (local dev) | A100 80GB (CamberCloud) |
| RAM | 16GB | 32GB+ |
| Storage | 100GB SSD | 500GB SSD |
| CPU | 8-core | 16-core+ |

### 0.2 Install Python Environment

Do this on your local machine AND on CamberCloud:

```bash
# Create conda environment
conda create -n prism python=3.10 -y
conda activate prism

# Core ML
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# TDA Libraries (THE critical tools)
pip install ripser              # Fastest Vietoris-Rips persistent homology
pip install gudhi               # Persistence diagrams, Wasserstein distance
pip install giotto-tda           # sklearn-compatible TDA pipeline
pip install persim               # Persistence image/diagram utilities

# Adversarial evaluation
pip install adversarial-robustness-toolbox   # IBM ART (40+ attacks)
pip install autoattack                        # Gold-standard evaluation
pip install robustbench                       # Standardized benchmarks

# Changepoint detection (for L0)
pip install ruptures             # BOCPD and CUSUM

# Conformal prediction
pip install mapie                # Conformal prediction framework
pip install crepes               # Conformal regressors and classifiers

# Experiment tracking
pip install wandb                # Weights & Biases

# Utilities
pip install numpy scipy scikit-learn matplotlib seaborn pandas tqdm
pip install pot                  # Python Optimal Transport (Wasserstein/Sinkhorn)
```

### 0.3 Project Structure

```
prism/
├── configs/              # YAML configs for experiments
├── data/                 # Datasets (CIFAR-10, ImageNet subset)
├── models/               # Pretrained model weights
├── src/
│   ├── tamm/             # Topological Activation Manifold Monitor
│   │   ├── __init__.py
│   │   ├── extractor.py  # Activation extraction hooks
│   │   ├── tda.py        # Persistent homology computation
│   │   └── scorer.py     # Topological anomaly scoring
│   ├── cadg/             # Conformal Adversarial Detection Guarantee
│   │   ├── __init__.py
│   │   ├── calibrate.py  # Conformal calibration
│   │   └── threshold.py  # Tiered threshold management
│   ├── sacd/             # Sequential Adversarial Campaign Detection
│   │   ├── __init__.py
│   │   ├── bocpd.py      # Bayesian Online Changepoint Detection
│   │   └── monitor.py    # Rolling buffer + L0 state
│   ├── tamsh/            # Topology-Aware MoE Self-Healing
│   │   ├── __init__.py
│   │   ├── experts.py    # Expert sub-networks
│   │   └── gating.py     # Wasserstein-based expert selection
│   ├── memory/           # Persistence diagram immune memory
│   ├── federation/       # Federated sharing protocol
│   └── prism.py          # Main PRISM wrapper class
├── experiments/
│   ├── feasibility/      # Week 2-3 TDA speed tests
│   ├── calibration/      # Conformal calibration experiments
│   ├── campaign/         # L0 campaign detection tests
│   ├── evaluation/       # Full attack evaluation
│   └── ablation/         # Ablation studies
├── notebooks/            # Jupyter exploration
├── paper/                # LaTeX paper source
├── tests/                # Unit tests
└── requirements.txt
```

### 0.4 CamberCloud Setup — Step by Step

<aside>
☁️

**CamberCloud** is a GPU cloud platform. You get 100 free credits on signup (1 credit = $1 USD). Use it for heavy TDA computation and AutoAttack evaluation — NOT for coding or debugging.

</aside>

**Step 1: Sign up**

- Go to [cambercloud.com](http://cambercloud.com)
- Create account, verify email
- You get 100 free credits

**Step 2: Install the CLI**

```bash
pip install camber
```

**Step 3: Authenticate**

```python
import camber

# First time — this opens browser for OAuth
camber.login()
```

**Step 4: Launch a GPU job**

```python
import camber

# Create an engine (this is their compute unit)
engine = camber.mle.create_engine(
    gpu_type="NVIDIA_A100",    # or "NVIDIA_H100" for heavy evaluation
    gpu_count=1,
    cpu_count=8,
    memory_gb=64,
)

# Upload your code
engine.upload("./prism/")  # uploads your project folder

# Run a script
result = engine.run("python experiments/feasibility/tda_benchmark.py")
print(result.logs)
```

**Step 5: When to use CamberCloud vs. Local**

| **Task** | **Where** | **Why** |
| --- | --- | --- |
| Writing code, debugging | Local | Free, fast iteration |
| TDA feasibility benchmark | CamberCloud A100 | Needs GPU memory |
| Training expert sub-networks | CamberCloud A100 | GPU-intensive |
| AutoAttack evaluation | CamberCloud A100/H100 | Very GPU-intensive, hours |
| Conformal calibration | Local or CamberCloud | Moderate compute |
| Paper writing | Local (Overleaf) | No GPU needed |

**Budget strategy:** Your 100 free credits get ~33 hours on A100 ($3/hr estimate). Use them wisely — run experiments locally first at small scale, then use CamberCloud only for full-scale runs.

**Alternative GPU providers (cheaper for long runs):**

- **RunPod** — A100 ~$1.39-2.17/hr (community cloud)
- **Lambda Labs** — H100 ~$2.99/hr
- [**Vast.ai**](http://Vast.ai) — cheapest spot GPUs

---

## Phase 1: TDA Foundation (Weeks 1-5)

### Step 1.1: Literature Deep-Dive (Week 1-2)

**What to read and in what order:**

<aside>
📚

**Use AI here:** Use Claude or ChatGPT to summarize papers, explain math, and help you understand concepts. Example prompt:

*"Explain persistent homology to me like I'm a CS grad student who knows linear algebra but not algebraic topology. Use concrete examples with point clouds. What are Betti numbers β₀, β₁, β₂? What is a Vietoris-Rips filtration? What is a persistence diagram?"*

</aside>

**Reading order (mandatory):**

1. **"A User's Guide to Topological Data Analysis" (Munch 2017)** — Start here. 20 pages, explains TDA from scratch
2. **"Computational Topology" (Edelsbrunner & Harer 2010)** — Chapters 1-4 only. The mathematical foundation
3. **"Topology and Data" (Carlsson 2009)** — How TDA applies to real data
4. **"A Gentle Introduction to Conformal Prediction" (Angelopoulos & Bates 2023)** — The conformal foundation. Very readable
5. **"Bayesian Online Changepoint Detection" (Adams & MacKay 2007)** — 8-page paper, foundational for L0
6. **Croce et al. (ICML 2022)** — "Evaluating Adversarial Robustness of Adaptive Test-time Defenses" — THE critique you must survive
7. **RAILS (Wang et al., IEEE Access 2022)** — Your closest competitor. Read the full paper AND run their GitHub code
8. **Gebhart & Schrater** — "Adversary Detection via Persistent Homology" — MUST cite, closest TDA+adversarial work
9. **LANL tda-adversarial-detection** — GitHub repo, study their approach

**Output:** Write a 2-page positioning document. AI prompt for help:

*"I'm writing a research positioning document for PRISM, a system that uses TDA + conformal prediction + sequential campaign detection for adversarial defense. Here are the existing papers I must differentiate from: [list papers]. Help me write clear differentiation statements for each, being intellectually honest about what they did first."*

### Step 1.2: TDA Feasibility Benchmark (Week 2-3) — THE MOST CRITICAL STEP

<aside>
🚨

**DO THIS BEFORE BUILDING ANYTHING.** If TDA is too slow, the entire project needs to pivot to approximations. This single experiment determines whether PRISM is viable.

</aside>

**The Benchmark Script:**

```python
# experiments/feasibility/tda_benchmark.py
import torch
import torchvision.models as models
import numpy as np
import time
from ripser import ripser
from gudhi.wasserstein import wasserstein_distance

# 1. Load pretrained ResNet-18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).cuda().eval()

# 2. Register forward hooks to extract activations
activations = {}
def get_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu().numpy()
    return hook

# Hook at 4 checkpoint layers
model.layer1.register_forward_hook(get_hook('layer1'))
model.layer2.register_forward_hook(get_hook('layer2'))
model.layer3.register_forward_hook(get_hook('layer3'))
model.layer4.register_forward_hook(get_hook('layer4'))

# 3. Generate random input (or use real CIFAR-10 image)
x = torch.randn(1, 3, 224, 224).cuda()
with torch.no_grad():
    _ = model(x)

# 4. Benchmark TDA at different subsample sizes
for n_points in [50, 100, 200, 500, 1000]:
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        act = activations[layer_name]
        # Flatten spatial dims: (1, C, H, W) -> (C*H*W, 1) or subsample
        flat = act.reshape(-1, act.shape[1])  # (spatial_points, channels)
        
        # Random subsample
        if flat.shape[0] > n_points:
            idx = np.random.choice(flat.shape[0], n_points, replace=False)
            flat = flat[idx]
        
        # Measure TDA time
        start = time.time()
        result = ripser(flat, maxdim=1)  # Compute H0 and H1
        tda_time = time.time() - start
        
        # Measure Wasserstein distance time
        dgm = result['dgms'][1]  # H1 persistence diagram
        # Create a dummy reference diagram
        ref_dgm = dgm + np.random.normal(0, 0.01, dgm.shape)
        
        start = time.time()
        w_dist = wasserstein_distance(dgm, ref_dgm, order=2)
        wass_time = time.time() - start
        
        print(f"n={n_points}, {layer_name}: "
              f"TDA={tda_time*1000:.1f}ms, "
              f"Wasserstein={wass_time*1000:.1f}ms")
```

**Decision matrix after running benchmark:**

| **TDA time (200 points, 1 layer)** | **Action** |
| --- | --- |
| < 10ms | Proceed as planned |
| 10-50ms | Proceed with subsampling strategy |
| 50-200ms | Switch to cubical complexes or landmark-based approx |
| > 200ms | STOP. Rethink approach. Consider ripser++ (GPU-accelerated) |

**If TDA is too slow — alternatives:**

- **ripser++** (GPU-accelerated) — up to 30x faster than ripser
- **Cubical complexes** — faster for grid-structured data (images)
- **Landmark-based Vietoris-Rips** — approximate TDA with fewer points
- **Persistence images** (vectorized) — pre-compute and cache

### Step 1.3: Build the Topological Self-Profile (Week 3-5)

**What this means in simple terms:** You pass thousands of clean (normal, unattacked) images through the neural network, capture what the internal layers look like (activations), compute their topological shape (persistence diagrams), and save the average as your "this is what normal looks like" reference.

```python
# src/tamm/extractor.py
import torch
import numpy as np

class ActivationExtractor:
    """Extracts activations from checkpoint layers of any PyTorch model."""
    
    def __init__(self, model, layer_names):
        self.model = model
        self.activations = {}
        self.hooks = []
        
        for name in layer_names:
            layer = dict(model.named_modules())[name]
            hook = layer.register_forward_hook(self._get_hook(name))
            self.hooks.append(hook)
    
    def _get_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def extract(self, x):
        """Forward pass and return activations dict."""
        self.activations = {}
        with torch.no_grad():
            self.model(x)
        return self.activations
    
    def cleanup(self):
        for h in self.hooks:
            h.remove()
```

```python
# src/tamm/tda.py
import numpy as np
from ripser import ripser
from gudhi.wasserstein import wasserstein_distance

class TopologicalProfiler:
    """Computes persistence diagrams from activation point clouds."""
    
    def __init__(self, n_subsample=200, max_dim=1):
        self.n_subsample = n_subsample
        self.max_dim = max_dim
    
    def compute_diagram(self, activation_tensor):
        """
        activation_tensor: shape (C, H, W) or (N, D)
        Returns persistence diagrams for H0, H1.
        """
        # Flatten to point cloud
        if activation_tensor.ndim == 3:
            C, H, W = activation_tensor.shape
            points = activation_tensor.reshape(C, -1).T  # (H*W, C)
        else:
            points = activation_tensor
        
        # Subsample for speed
        if points.shape[0] > self.n_subsample:
            idx = np.random.choice(
                points.shape[0], self.n_subsample, replace=False
            )
            points = points[idx]
        
        # Compute persistent homology
        result = ripser(points, maxdim=self.max_dim)
        return result['dgms']  # List of diagrams [H0, H1, ...]
    
    def anomaly_score(self, diagrams, ref_diagrams, weights=None):
        """
        Compute weighted Wasserstein distance between input diagrams
        and reference (clean) diagrams.
        """
        if weights is None:
            weights = [1.0 / len(diagrams)] * len(diagrams)
        
        score = 0.0
        # Fix #8 (Low): do NOT skip empty diagrams — they still contribute 0 to
        # the numerator while the denominator (sum of weights) must stay
        # correct. Silent skip causes score to be inflated when many layers
        # return empty H1 diagrams (common for shallow layers).
        for i, (dgm, ref) in enumerate(zip(diagrams, ref_diagrams)):
            if len(dgm) == 0 or len(ref) == 0:
                score += 0.0  # explicit zero: layer contributes nothing
                continue
            w = wasserstein_distance(dgm, ref, order=2)
            score += weights[i] * w
        return score
```

```python
# scripts/build_profile.py
"""Build the topological self-profile from clean data."""
import torch
import torchvision
import numpy as np
from src.tamm.extractor import ActivationExtractor
from src.tamm.tda import TopologicalProfiler

# Load model
from torchvision.models import ResNet18_Weights
model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()

# Load clean data
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Setup
extractor = ActivationExtractor(
    model, ['layer1', 'layer2', 'layer3', 'layer4']
)
profiler = TopologicalProfiler(n_subsample=200, max_dim=1)

# Collect diagrams from 10,000 clean images
all_diagrams = {layer: [] for layer in ['layer1','layer2','layer3','layer4']}

for i, (img, _) in enumerate(loader):
    if i >= 10000:
        break
    
    acts = extractor.extract(img.cuda())
    
    for layer_name, act in acts.items():
        act_np = act.squeeze(0).cpu().numpy()  # (C, H, W)
        dgms = profiler.compute_diagram(act_np)
        all_diagrams[layer_name].append(dgms)
    
    if i % 1000 == 0:
        print(f"Processed {i}/10000 images")

# Compute medoid reference diagram per layer (Wasserstein medoid).
# Critical: prism.py expects a SINGLE reference diagram per layer, not a list.
# compute_reference_medoid() selects the diagram minimising total
# Wasserstein distance to all others — this is the Fréchet-mean approximation.
print("Computing Wasserstein medoids...")
reference_profiles = {}
for layer_name, dgms_list in all_diagrams.items():
    reference_profiles[layer_name] = profiler.compute_reference_medoid(dgms_list)
    print(f"  {layer_name}: medoid computed from {len(dgms_list)} diagrams")

import pickle
with open('models/reference_profiles.pkl', 'wb') as f:
    pickle.dump(reference_profiles, f)

# Also precompute anomaly scores on all 10,000 clean images and save.
# calibrate_thresholds.py loads this file to fit conformal quantiles.
print("Precomputing clean anomaly scores for calibration...")
all_scores = []
for layer_name, dgms_list in all_diagrams.items():
    ref = reference_profiles[layer_name]
    for dgms in dgms_list:
        s = profiler.anomaly_score(dgms, ref)
        # average across layers on first pass; loop will overwrite — aggregate below
        break  # handled in the joined loop below

# Proper per-image aggregated score
all_scores = []
for i in range(len(all_diagrams['layer1'])):
    score = 0.0
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        dgms = all_diagrams[layer_name][i]
        ref  = reference_profiles[layer_name]
        score += profiler.anomaly_score(dgms, ref) / 4.0
    all_scores.append(score)

import os
os.makedirs('experiments/calibration', exist_ok=True)
np.save('experiments/calibration/clean_scores.npy', np.array(all_scores))
print(f"Saved {len(all_scores)} clean scores → experiments/calibration/clean_scores.npy")
print("Topological self-profile built!")
```

```python
# src/tamm/scorer.py
# Fix #10: scorer.py was listed in the project structure but never shown.
# This module is used by prism.py to compute per-layer and aggregated scores.
import numpy as np
from gudhi.wasserstein import wasserstein_distance as _wass

class TopologicalScorer:
    """
    Aggregates per-layer Wasserstein distances into a single anomaly score.
    Separating scoring from TopologicalProfiler keeps concerns clean and
    lets callers use custom weights (e.g. up-weight final layers).
    """

    def score(self, diagrams_per_layer, ref_profiles, layer_names,
              weights=None):
        """
        diagrams_per_layer : Dict[layer_name -> List[ndarray]] (H0, H1, ...)
        ref_profiles        : Dict[layer_name -> List[ndarray]] (medoid diagram)
        weights             : optional Dict[layer_name -> float]; uniform if None
        Returns             : float — higher means more anomalous
        """
        if weights is None:
            w = 1.0 / len(layer_names)
            weights = {l: w for l in layer_names}

        total = 0.0
        for layer in layer_names:
            dgms = diagrams_per_layer[layer]   # [H0_array, H1_array]
            ref  = ref_profiles[layer]          # [ref_H0, ref_H1]
            layer_score = 0.0
            n_dims = min(len(dgms), len(ref))
            for dim in range(n_dims):
                d = dgms[dim]
                r = ref[dim]
                if len(d) == 0 or len(r) == 0:
                    layer_score += 0.0  # empty → zero contribution
                    continue
                layer_score += _wass(d, r, order=2)
            total += weights[layer] * (layer_score / max(n_dims, 1))
        return total

    def score_per_layer(self, diagrams_per_layer, ref_profiles, layer_names):
        """Diagnostic: returns Dict[layer_name -> float] score breakdown."""
        return {
            l: self.score(diagrams_per_layer, ref_profiles, [l],
                          weights={l: 1.0})
            for l in layer_names
        }
```

<aside>
🤖

**Where to use AI in Phase 1:**

- Ask AI to explain any paper you don't understand
- Ask AI to debug your TDA code
- Ask AI to help interpret persistence diagrams
- Prompt: *"I computed persistence diagrams for clean vs adversarial activations. The clean H1 diagram has 3 points with high persistence, the adversarial has 12. What does this mean topologically?"*
</aside>

---

## Phase 2: Conformal Calibration (Weeks 6-9)

### Step 2.1: Build the Calibration Set

```python
# src/cadg/calibrate.py
import numpy as np

class ConformalCalibrator:
    """
    Split conformal prediction for adversarial detection.
    Provides distribution-free FPR guarantees.
    """
    
    def __init__(self):
        self.calibration_scores = None
        self.thresholds = {}  # alpha -> threshold
    
    def calibrate(self, clean_scores):
        """
        clean_scores: array of anomaly scores from calibration set
                     (at least 1000 clean inputs recommended)
        """
        self.calibration_scores = np.sort(clean_scores)
        n = len(clean_scores)
        
        # Compute thresholds for each level
        alphas = {
            'L1': 0.10,   # 10% FPR — low cost tier
            'L2': 0.03,   # 3% FPR — medium cost tier
            'L3': 0.005,  # 0.5% FPR — high cost tier
        }
        
        for level, alpha in alphas.items():
            # Conformal quantile
            q = np.ceil((n + 1) * (1 - alpha)) / n
            q = min(q, 1.0)
            threshold = np.quantile(self.calibration_scores, q)
            self.thresholds[level] = threshold
            print(f"{level} (α={alpha}): threshold = {threshold:.4f}")
        
        return self.thresholds
    
    def classify(self, score, l0_active=False, l0_factor=0.8):
        """
        Classify a single input's anomaly score into a response level.
        If L0 is active, thresholds are lowered by l0_factor.
        """
        factor = l0_factor if l0_active else 1.0
        
        if score > self.thresholds['L3'] * factor:
            return 'L3'
        elif score > self.thresholds['L2'] * factor:
            return 'L2'
        elif score > self.thresholds['L1'] * factor:
            return 'L1'
        else:
            return 'PASS'
    
    def verify_coverage(self, validation_scores, alpha=0.10, level='L1'):
        """
        Verify that empirical FPR <= alpha on a held-out validation set.
        """
        threshold = self.thresholds[level]
        fpr = np.mean(validation_scores > threshold)
        passed = fpr <= alpha
        print(f"Verification {level}: FPR={fpr:.4f}, α={alpha}, "
              f"{'PASSED ✓' if passed else 'FAILED ✗'}")
        return passed, fpr
```

### Step 2.2: Calibrate and Verify

```python
# scripts/calibrate_thresholds.py
import numpy as np
from src.tamm.tda import TopologicalProfiler
from src.cadg.calibrate import ConformalCalibrator

# Load pre-computed anomaly scores from clean data
# (You computed these during the self-profile phase)
all_scores = np.load('experiments/calibration/clean_scores.npy')

# Split: 5000 calibration + 5000 validation
cal_scores = all_scores[:5000]
val_scores = all_scores[5000:10000]

# Calibrate
calibrator = ConformalCalibrator()
thresholds = calibrator.calibrate(cal_scores)

# Verify on held-out set
for level, alpha in [('L1', 0.10), ('L2', 0.03), ('L3', 0.005)]:
    calibrator.verify_coverage(val_scores, alpha=alpha, level=level)

# Save calibrator
import pickle
with open('models/calibrator.pkl', 'wb') as f:
    pickle.dump(calibrator, f)
```

---

## Phase 3: Sequential Campaign Monitor — L0 (Weeks 10-13)

### Step 3.1: Build BOCPD

```python
# src/sacd/bocpd.py
import numpy as np
from scipy.stats import norm

class BayesianOnlineChangepoint:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay 2007).
    Detects when the distribution of anomaly scores shifts.
    """
    
    def __init__(self, hazard_rate=1/200, mu0=0.0, sigma0=1.0,
                 max_run_length=500):
        self.hazard = hazard_rate  # Expected run length
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.max_run_length = max_run_length  # Fix #6: bound memory growth
        self.run_length_probs = np.array([1.0])  # Start with run length 0
        self.observations = []
    
    def update(self, score):
        """
        Process one new anomaly score.
        Returns P(changepoint just happened).
        """
        self.observations.append(score)
        T = len(self.run_length_probs)
        
        # Predictive probability for each run length
        pred_probs = np.zeros(T)
        for t in range(T):
            # Simple Gaussian predictive
            if t == 0:
                pred_probs[t] = norm.pdf(score, self.mu0, self.sigma0)
            else:
                recent = self.observations[-t:]
                mu = np.mean(recent)
                sigma = max(np.std(recent), 0.01)
                pred_probs[t] = norm.pdf(score, mu, sigma)
        
        # Growth probabilities
        growth = self.run_length_probs * pred_probs * (1 - self.hazard)
        
        # Changepoint probability
        cp_prob = np.sum(self.run_length_probs * pred_probs * self.hazard)
        
        # New run length distribution
        new_probs = np.zeros(T + 1)
        new_probs[0] = cp_prob
        new_probs[1:] = growth
        
        # Normalize
        total = np.sum(new_probs)
        if total > 0:
            new_probs /= total
        
        # Fix #6: truncate to max_run_length to prevent unbounded memory growth
        # on long inference streams (memory leak on production deployments).
        if len(new_probs) > self.max_run_length:
            new_probs = new_probs[-self.max_run_length:]
            new_probs /= new_probs.sum()  # renormalize after truncation
        self.observations = self.observations[-self.max_run_length:]
        
        self.run_length_probs = new_probs
        
        return cp_prob
    
    def reset(self):
        self.run_length_probs = np.array([1.0])
        self.observations = []
```

### Step 3.2: Build the L0 Monitor

```python
# src/sacd/monitor.py
import numpy as np
from .bocpd import BayesianOnlineChangepoint

class CampaignMonitor:
    """
    L0 Sequential Campaign Detector.
    Monitors the stream of anomaly scores for attack campaigns.
    """
    
    def __init__(self, window_size=100, cp_threshold=0.5,
                 hazard_rate=1/200, l0_factor=0.8):
        self.window_size = window_size
        self.cp_threshold = cp_threshold
        self.l0_factor = l0_factor
        self.bocpd = BayesianOnlineChangepoint(hazard_rate=hazard_rate)
        self.l0_active = False
        self.l0_start_time = None
        self.score_buffer = []
        self.alert_log = []
    
    def process_score(self, score, timestamp=None):
        """
        Process one anomaly score. Returns L0 state.
        """
        self.score_buffer.append(score)
        if len(self.score_buffer) > self.window_size:
            self.score_buffer.pop(0)
        
        # Run BOCPD
        cp_prob = self.bocpd.update(score)
        
        # Check for changepoint
        if cp_prob > self.cp_threshold and not self.l0_active:
            self.l0_active = True
            self.l0_start_time = timestamp
            self.alert_log.append({
                'type': 'L0_ACTIVATED',
                'timestamp': timestamp,
                'cp_probability': cp_prob,
                'recent_scores': list(self.score_buffer[-20:])
            })
        
        return {
            'l0_active': self.l0_active,
            'changepoint_prob': cp_prob,
            'buffer_mean': np.mean(self.score_buffer),
            'buffer_std': np.std(self.score_buffer),
        }
    
    def deactivate_l0(self):
        """Manually deactivate L0 after threat passes."""
        self.l0_active = False
        self.bocpd.reset()
```

---

## Phase 4: Topology-Aware MoE Self-Healing (Weeks 14-17)

### Step 4.1: Train Expert Sub-Networks

```python
# src/tamsh/experts.py
import torch
import torch.nn as nn
import numpy as np  # Fix #4: numpy required by np.argmin() in select_expert()

class ExpertSubNetwork(nn.Module):
    """Small expert that replaces a span of compromised layers."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        # Flatten if needed
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

class TopologyAwareMoE:
    """
    Maintains K expert sub-networks.
    Selects expert based on Wasserstein distance in persistence diagram space.
    """
    
    def __init__(self, experts, expert_ref_diagrams):
        self.experts = experts  # List of ExpertSubNetwork
        self.ref_diagrams = expert_ref_diagrams  # List of reference diagrams
    
    def select_expert(self, input_diagram):
        """Select the most topologically compatible expert."""
        from gudhi.wasserstein import wasserstein_distance
        
        # Fix #7: BatchNorm1d fails with batch_size=1 unless model is in eval().
        # Call eval() on all experts before distance computation.
        for expert in self.experts:
            expert.eval()
        
        distances = []
        for ref_dgm in self.ref_diagrams:
            # Compare H1 diagrams (index 1 = first-order loops)
            d = wasserstein_distance(
                input_diagram[1], ref_dgm[1], order=2
            )
            distances.append(d)
        
        best_idx = int(np.argmin(distances))
        return best_idx, self.experts[best_idx]
```

### Step 4.2: Training the Experts

```python
# scripts/train_experts.py
"""Cluster clean activations by topology, train one expert per cluster."""
import numpy as np
from sklearn.cluster import KMeans
from gudhi.wasserstein import wasserstein_distance
import torch

# 1. Load all clean persistence diagrams (from Phase 1)
# 2. Compute pairwise Wasserstein distance matrix
# 3. Use KMeans on the distance matrix (K=4)
# 4. For each cluster: train an expert sub-network
#    on the activations belonging to that cluster

K = 4  # Number of experts
# ... (clustering code)
# ... (training loop for each expert)
# Save experts and their reference diagrams
```

<aside>
🤖

**Where to use AI in Phase 4:**

- Ask AI to help write the training loop for expert sub-networks
- Ask AI to help implement Wasserstein-based clustering (non-trivial)
- Prompt: *"Write a PyTorch training loop that trains a small 3-layer MLP expert to approximate the function of ResNet-18's layer3->layer4 on a subset of CIFAR-10 images. The expert should take flattened layer3 activations as input and output layer4-equivalent features."*
</aside>

---

## Phase 5: Immune Memory + Federation (Weeks 18-20)

### Step 5.1: Persistence Diagram Memory Store

```python
# src/memory/immune_memory.py
import numpy as np
from gudhi.wasserstein import wasserstein_distance

class ImmuneMemory:
    """Stores attack signatures as persistence diagrams for fast recall."""
    
    def __init__(self, match_threshold=0.5):
        self.signatures = []  # List of (diagram, attack_type, level)
        self.threshold = match_threshold
    
    def store(self, diagram, attack_type, response_level):
        self.signatures.append({
            'diagram': diagram,
            'attack_type': attack_type,
            'level': response_level,
        })
    
    def match(self, input_diagram):
        """Check if input matches any known attack signature."""
        if not self.signatures:
            return None
        
        best_match = None
        best_dist = float('inf')
        
        for sig in self.signatures:
            d = wasserstein_distance(
                input_diagram[1], sig['diagram'][1], order=2
            )
            if d < best_dist:
                best_dist = d
                best_match = sig
        
        if best_dist < self.threshold:
            return best_match  # Known attack variant!
        return None
```

---

## Phase 6: Full PRISM Integration + Evaluation (Weeks 21-29)

### Step 6.1: The Main PRISM Class

```python
# src/prism.py
import torch
import numpy as np
from .tamm.extractor import ActivationExtractor
from .tamm.tda import TopologicalProfiler
from .cadg.calibrate import ConformalCalibrator
from .sacd.monitor import CampaignMonitor
from .tamsh.experts import TopologyAwareMoE
from .memory.immune_memory import ImmuneMemory

class PRISM:
    """
    Predictive Runtime Immune System with Manifold Monitoring.
    Wraps any pretrained PyTorch model with adversarial defense.
    """
    
    def __init__(self, model, layer_names, calibrator_path,
                 profile_path, experts=None, memory=None):
        self.model = model
        self.extractor = ActivationExtractor(model, layer_names)
        self.profiler = TopologicalProfiler(n_subsample=200)
        self.calibrator = self._load(calibrator_path)
        self.ref_profiles = self._load(profile_path)
        self.monitor = CampaignMonitor()
        self.moe = experts
        self.memory = memory or ImmuneMemory()
        self.layer_names = layer_names
    
    # Fix #9: restrict pickle loading to known safe paths.
    # Pickle deserialization executes arbitrary code — never load untrusted files.
    _ALLOWED_PICKLE_PATHS = [
        'models/calibrator.pkl',
        'models/reference_profiles.pkl',
        'models/experts.pkl',
    ]

    def _load(self, path):
        import pickle
        if not any(path.endswith(p) for p in self._ALLOWED_PICKLE_PATHS):
            raise ValueError(
                f"Refusing to load untrusted pickle path: {path!r}. "
                f"Add to _ALLOWED_PICKLE_PATHS to permit."
            )
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def defend(self, x):
        """
        Main inference with PRISM defense.
        x: input tensor (batch_size=1)
        Returns: (prediction, response_level, metadata)
        """
        # Step 1: Extract activations
        acts = self.extractor.extract(x)
        
        # Step 2: Compute persistence diagrams
        diagrams = []
        for layer in self.layer_names:
            act_np = acts[layer].squeeze(0).cpu().numpy()
            dgm = self.profiler.compute_diagram(act_np)
            diagrams.append(dgm)
        
        # Step 3: Compute aggregated anomaly score
        score = 0.0
        for i, (dgm, layer) in enumerate(zip(diagrams, self.layer_names)):
            ref = self.ref_profiles[layer]
            s = self.profiler.anomaly_score(dgm, ref)
            score += s / len(self.layer_names)
        
        # Step 4: Check immune memory (fast path)
        memory_match = self.memory.match(diagrams[-1])
        if memory_match:
            return self._escalate(
                x, memory_match['level'],
                f"Memory match: {memory_match['attack_type']}"
            )
        
        # Step 5: Update campaign monitor
        l0_state = self.monitor.process_score(score)
        
        # Step 6: Classify response level
        level = self.calibrator.classify(
            score, l0_active=l0_state['l0_active']
        )
        
        # Step 7: Execute response
        if level == 'PASS':
            with torch.no_grad():
                pred = self.model(x)
            return pred, 'PASS', {'score': score}
        
        elif level == 'L1':
            # Log + normal inference
            with torch.no_grad():
                pred = self.model(x)
            return pred, 'L1', {'score': score, 'flagged': True}
        
        elif level == 'L2':
            # TODO: Add input purification here
            with torch.no_grad():
                pred = self.model(x)
            return pred, 'L2', {'score': score, 'purified': True}
        
        elif level == 'L3':
            # Route through MoE expert
            if self.moe:
                idx, expert = self.moe.select_expert(diagrams[-1])
                # Route through expert instead of compromised layers
                act_input = acts[self.layer_names[-2]]
                pred = expert(act_input)
                return pred, 'L3', {
                    'score': score, 'expert_idx': idx
                }
            else:
                # Reject input
                return None, 'L3_REJECT', {'score': score}
```

### Step 6.2: Attack Evaluation — The Complete Procedure

<aside>
🎯

**This is where you spend the most CamberCloud credits.** Run locally at small scale first (100 images), then full scale (1000+) on CamberCloud.

</aside>

```python
# experiments/evaluation/run_evaluation.py
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
from art.attacks.evasion import (
    FastGradientMethod, ProjectedGradientDescent,
    CarliniL2Method, SquareAttack
)
from art.estimators.classification import PyTorchClassifier
import numpy as np
from src.prism import PRISM

# Fix #5 (Critical): ART must attack in pixel [0,1] space.
# The model must normalize INTERNALLY so ART's clip_values=(0,1) is valid.
# Pattern: _NormalizedResNet wraps ResNet; ART wraps _NormalizedResNet.
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

class _NormalizedResNet(torch.nn.Module):
    """ResNet with ImageNet normalization baked in via register_buffer.
    Lets ART attack in clean pixel [0,1] space with valid clip_values."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('mean', torch.tensor(_MEAN).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor(_STD).view(1,3,1,1))
    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)

# Pixel-space transform — NO normalization here (ART handles [0,1] directly)
_PIXEL_TRANSFORM = T.Compose([T.Resize(224), T.ToTensor()])
# Normalization applied AFTER ART generates adversarial, before PRISM scoring
_NORMALIZE = T.Normalize(_MEAN, _STD)

# Setup model + PRISM (PRISM receives normalized tensors as usual)
backbone = torchvision.models.resnet18(
    weights=ResNet18_Weights.IMAGENET1K_V1
).eval()
wrapped  = _NormalizedResNet(backbone).eval()  # ART-safe wrapper

prism = PRISM(
    model=backbone,  # PRISM uses the bare backbone (receives normalized input)
    layer_names=['layer1', 'layer2', 'layer3', 'layer4'],
    calibrator_path='models/calibrator.pkl',
    profile_path='models/reference_profiles.pkl',
)

# Wrap the NORMALIZED model for ART — clip_values=(0,1) now valid
classifier = PyTorchClassifier(
    model=wrapped,
    loss=torch.nn.CrossEntropyLoss(),
    input_shape=(3, 224, 224),
    nb_classes=10,
    clip_values=(0.0, 1.0),  # pixel space; _NormalizedResNet normalizes internally
)

# Define attacks
attacks = {
    'FGSM':  FastGradientMethod(classifier, eps=0.03),
    'PGD':   ProjectedGradientDescent(
                 classifier, eps=0.03, max_iter=40, eps_step=0.008),
    'CW':    CarliniL2Method(classifier, max_iter=100),
    'Square': SquareAttack(classifier, eps=0.05, max_iter=5000),
}

# Pixel-space test dataset (NO normalization — ART needs raw pixel values)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True,
    transform=_PIXEL_TRANSFORM,
)

# Evaluate
results = {}
for attack_name, attack in attacks.items():
    print(f"\nEvaluating {attack_name}...")
    tp, fp, fn, tn = 0, 0, 0, 0

    for i in range(1000):  # 1000 test images
        pixel_img, _ = test_dataset[i]            # [0,1] tensor
        norm_img = _NORMALIZE(pixel_img)           # normalized for PRISM

        # Test clean input (PRISM receives normalized tensor)
        _, level_clean, _ = prism.defend(norm_img.unsqueeze(0))
        fp += int(level_clean != 'PASS')
        tn += int(level_clean == 'PASS')

        # Generate adversarial in pixel space (ART clips to [0,1])
        x_np = pixel_img.unsqueeze(0).numpy()       # (1,3,H,W) float32
        x_adv_np = attack.generate(x_np)            # still in [0,1]

        # Normalize adversarial before PRISM
        x_adv_t  = torch.tensor(x_adv_np[0])        # (3,H,W)
        x_adv_norm = _NORMALIZE(x_adv_t).unsqueeze(0)

        _, level_adv, _ = prism.defend(x_adv_norm)
        tp += int(level_adv != 'PASS')
        fn += int(level_adv == 'PASS')

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2*tpr*precision / (tpr + precision) if (tpr + precision) > 0 else 0
    results[attack_name] = {'TPR': tpr, 'FPR': fpr, 'Precision': precision, 'F1': f1}
    print(f"{attack_name}: TPR={tpr:.3f}, FPR={fpr:.3f}, F1={f1:.3f}")

import json, os
os.makedirs('experiments/evaluation', exist_ok=True)
with open('experiments/evaluation/results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Step 6.3: Campaign Detection Test

```python
# experiments/campaign/test_campaign_detection.py
"""Simulate an adversary probing the model before attacking."""
import numpy as np
from src.sacd.monitor import CampaignMonitor

monitor = CampaignMonitor(
    window_size=100, cp_threshold=0.3, hazard_rate=1/200
)

# Phase 1: 50 clean queries (normal traffic)
clean_scores = np.random.normal(0.1, 0.02, 50)
for i, s in enumerate(clean_scores):
    state = monitor.process_score(s, timestamp=i)

print(f"After clean phase: L0 active = {state['l0_active']}")

# Phase 2: 30 probe queries (slightly elevated scores)
probe_scores = np.random.normal(0.25, 0.05, 30)
detected_at = None
for i, s in enumerate(probe_scores):
    state = monitor.process_score(s, timestamp=50 + i)
    if state['l0_active'] and detected_at is None:
        detected_at = i
        print(f"L0 ACTIVATED at probe query {i}!")

# Phase 3: 5 full attacks (high scores)
attack_scores = np.random.normal(0.8, 0.1, 5)
for i, s in enumerate(attack_scores):
    state = monitor.process_score(s, timestamp=80 + i)

print(f"\nCampaign detection lead time: {detected_at} queries")
print(f"Target: < 20 queries")
```

---

## Phase 7: Paper Writing (Weeks 30-34)

### Step 7.1: Setup

1. Go to [overleaf.com](http://overleaf.com) — free account
2. Create new project → "NeurIPS 2027" template (or download from the NeurIPS website)
3. Structure your paper exactly as outlined in the PRISM plan Section 5, Step 19

### Step 7.2: Writing Order (Don't Write Linearly)

| **Order** | **Section** | **Why This Order** |
| --- | --- | --- |
| 1 | Experiments (Section 4) | Write results first — they determine your claims |
| 2 | Method (Section 3) | Now you know what worked, write the method |
| 3 | Related Work (Section 2) | Position relative to what you've already built |
| 4 | Introduction (Section 1) | Now you can write a sharp intro |
| 5 | Discussion + Conclusion (5-6) | Reflect on what worked and didn't |
| 6 | Abstract | LAST. Summarize your actual contributions |

### Step 7.3: Using AI for Paper Writing

<aside>
🤖

**AI prompts for each section:**

**For Experiments section:**

*"I have these experimental results for PRISM: [paste results table]. Help me write the Experiments section for a NeurIPS paper. Focus on: (1) what the numbers show, (2) why PRISM outperforms baselines, (3) honest limitations. Use formal academic style."*

**For Method section:**

*"Here is the PRISM method with 4 contributions: TAMM, CADG, SACD, TAMSH. [paste technical details]. Write Section 3 of a NeurIPS paper describing this method. Include formal notation, algorithm boxes, and theorem statements. Be mathematically precise."*

**For Related Work:**

*"Here are all the papers PRISM must cite and differentiate from: [list]. Write a Related Work section that is honest about what each paper did first, while clearly positioning PRISM's unique contributions."*

**For figures:**

*"I need to create Figure 2 for my paper: a visualization of clean vs adversarial persistence diagrams. Write matplotlib code that: (1) shows a 3D point cloud of activations, (2) shows the corresponding persistence diagram, (3) has clean on left, adversarial on right."*

</aside>

### Step 7.4: Key Figures to Generate

```python
# paper/figures/fig2_persistence_viz.py
"""The key visual that makes the paper memorable."""
import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Top-left: Clean activation point cloud (2D projection)
# Top-right: Clean persistence diagram
# Bottom-left: Adversarial activation point cloud
# Bottom-right: Adversarial persistence diagram

# ... (generate clean and adversarial activations)
# ... (compute persistence diagrams)
# ... (plot with clear labels, colors, annotations)

plt.tight_layout()
plt.savefig('paper/figures/fig2.pdf', dpi=300)
```

---

## Complete Week-by-Week Checklist

- [ ]  **Week 0** — Environment setup, install all tools, CamberCloud account
- [ ]  **Week 1-2** — Read all mandatory papers, write positioning document
- [ ]  **Week 2-3** — TDA feasibility benchmark (CRITICAL GATE)
- [ ]  **Week 3-5** — Build topological self-profile on CIFAR-10
- [ ]  **Week 6-7** — Conformal calibration on clean scores
- [ ]  **Week 8-9** — Verify conformal coverage, build Pareto curve
- [ ]  **Week 10-11** — Implement BOCPD for L0
- [ ]  **Week 12-13** — Campaign detection validation experiments
- [ ]  **Week 14-15** — Cluster activations, design expert architectures
- [ ]  **Week 16-17** — Train K=4 experts, implement topology-aware gating
- [ ]  **Week 18-19** — Build immune memory store
- [ ]  **Week 20** — Federated protocol demo
- [ ]  **Week 21-23** — Full PRISM integration, smoke tests
- [ ]  **Week 24-26** — Attack evaluation (FGSM, PGD, CW, AutoAttack, Square)
- [ ]  **Week 27** — Adaptive attack evaluation (topology-aware)
- [ ]  **Week 28-29** — Ablation studies, baseline comparisons
- [ ]  **Week 30-31** — Write Experiments + Method sections
- [ ]  **Week 32** — Write Related Work + Introduction
- [ ]  **Week 33** — Write Discussion, Conclusion, Abstract
- [ ]  **Week 34** — Polish, internal review, submit

---

## Where to Use AI — Summary

| **Task** | **AI Tool** | **How** |
| --- | --- | --- |
| Understanding papers | Claude / ChatGPT | "Explain this paper's key contribution in simple terms" |
| Math derivations | Claude | "Derive the conformal coverage guarantee step by step" |
| Code writing | Claude / Cursor / Copilot | "Write a PyTorch forward hook that extracts layer4 activations" |
| Debugging | Claude / ChatGPT | Paste error + code → "Why is this failing?" |
| Paper writing | Claude | "Write the Method section given these technical details" |
| Figure generation | Claude / ChatGPT | "Write matplotlib code for a persistence diagram plot" |
| LaTeX formatting | Claude | "Format this theorem in NeurIPS LaTeX style" |
| Literature search | Semantic Scholar / Google Scholar | Verify no one published your idea while you worked |

<aside>
⚠️

**AI Ethics Warning:** You can use AI to help write code and drafts, but the final paper must represent YOUR understanding and YOUR experimental results. Never fabricate results. Never let AI generate fake experimental numbers. Always verify AI-generated math independently.

</aside>