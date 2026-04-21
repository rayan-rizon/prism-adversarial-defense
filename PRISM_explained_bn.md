# PRISM গবেষণাপত্রের পূর্ণাঙ্গ বাংলা ব্যাখ্যা

## ভূমিকা

এই ডকুমেন্টটি PRISM গবেষণাপত্রের সম্পূর্ণ, সহজবোধ্য বাংলা ব্যাখ্যা। এখানে পেপারটি কী করছে, কেন করছে, কীভাবে করছে, কী ধরনের গণিত ব্যবহার করছে, FGSM/PGD/CW/Square/AutoAttack/FPR/TPR/Wasserstein/ResNet ইত্যাদি term-এর মানে কী, এবং সংখ্যাগুলো কী বোঝায়, সব এক জায়গায় সাজানো হয়েছে।

গুরুত্বপূর্ণ নোট: পেপারের manuscript-এর কিছু সংখ্যার সঙ্গে verified local artifacts-এর কিছু পার্থক্য আছে। তাই যেখানে সম্ভব, আমি verified artifact-এর সংখ্যাগুলোকে বেশি নির্ভরযোগ্য ধরেছি।

## 1. PRISM-এর মূল উদ্দেশ্য

PRISM-এর পূর্ণ নাম: **Predictive Runtime Immune System with Manifold Monitoring**।

এটার মূল উদ্দেশ্য হলো adversarial example detection।

সহজ ভাষায়:
- একটা neural network সাধারণ clean image-এ ঠিকভাবে কাজ করে।
- কিন্তু adversarial image এমনভাবে বানানো হয় যাতে মানুষ প্রায় বুঝতে না পারে, কিন্তু model ভুল prediction দেয়।
- PRISM model-এর output দেখার বদলে model-এর ভেতরের activation বা internal representation দেখে বুঝতে চায় inputটা clean নাকি suspicious।

অর্থাৎ PRISM একটি **runtime defense**:
- model train করার সময় defensive retraining করতে হয় না,
- inference চলার সময় attack detect করার চেষ্টা করে,
- detected attack হলে reaction tier অনুযায়ী ভিন্ন ভিন্ন action নেয়।

## 2. পেপারের বড় ধারণা

PRISM চারটি component একসাথে ব্যবহার করে:

1. **TAMM** - Topological Activation Manifold Monitor
2. **CADG** - Conformal Adversarial Detection Guarantee
3. **SACD** - Sequential Adversarial Campaign Detection
4. **TAMSH** - Topology-Aware MoE Self-Healing

এগুলো মিলিয়ে পুরো pipeline এমন:

- input image network-এ যায়
- কয়েকটি hidden layer-এর activation বের হয়
- activation কে point cloud হিসেবে ধরা হয়
- সেই point cloud-এর topology বের করা হয় persistent homology দিয়ে
- topology summary হিসেবে persistence diagram পাওয়া যায়
- clean reference diagram-এর সঙ্গে Wasserstein distance মাপা হয়
- distance বেশি হলে inputকে unusual/suspicious ধরা হয়
- clean calibration data ব্যবহার করে threshold ঠিক করা হয়
- online query stream-এ যদি attack campaign শুরু হয়, তাহলে threshold আরও sensitive করা হয়
- খুব high-severity case-এ input hard reject না করে expert subnet-এ পাঠানো হয়

## 3. কেন topology?

PRISM-এর key insight হলো:
- clean input আর adversarial input network-এর ভিতরে একই ধরনের পথ অনুসরণ করে না
- intermediate feature space-এর shape বদলে যায়
- সেই shape বা topology ধরা গেলে attack detection সম্ভব

এখানে topology বলতে geometric shape-এর মতো কিছু বোঝানো হয়, কিন্তু ordinary geometry না; বরং connected component, loop, hole ইত্যাদি abstract structure।

## 4. গবেষণাপত্রের কাঠামো

পেপারটি মূলত এই অংশগুলো নিয়ে তৈরি:
- Introduction
- Related Work
- Method
- Experiments
- Conclusion
- Appendix

এখন একে একে বিস্তারিত বোঝানো হলো।

---

## 5. Base model: ResNet-18 কী

পেপারে base classifier হিসেবে **ResNet-18** ব্যবহার করা হয়েছে।

ResNet মানে Residual Network।

এটা কেন গুরুত্বপূর্ণ:
- deep network-এ gradient vanish সমস্যা কমাতে residual connection ব্যবহার হয়
- skip connection model-কে easier optimization দেয়
- CIFAR-10 বা ImageNet-এর মতো vision task-এ এটি বহুল ব্যবহৃত

PRISM ResNet-18-এর output classify করার বদলে তার মধ্যবর্তী layer-এর activation monitor করে।

পেপারে monitored layers হিসেবে layer2, layer3, layer4 ব্যবহার করা হয়েছে।

## 6. Layer কী

Layer হলো neural network-এর একটি processing stage।

উদাহরণ:
- প্রথম layer edge-এর মতো simple feature ধরতে পারে
- মাঝের layer texture/pattern ধরতে পারে
- deep layer object-level semantics ধরতে পারে

PRISM-এর ধারণা হলো, attack যদি image-level-এ subtle হয়, internal layer-এ তার effect অনেক বড় বা structured হতে পারে।

## 7. Activation কী

Activation হলো কোনো layer-এর output tensor।

এটাকে সহজে feature map বলা যায়।

ধারণা:
- image input → conv layer → activation
- activation network-এর learned representation দেখায়

PRISM এই activation tensor থেকে topology বের করে।

## 8. TDA, persistent homology, persistence diagram

### 8.1 TDA কী

TDA মানে **Topological Data Analysis**।

এটা data-এর shape বা structure study করার কৌশল।

### 8.2 Persistent homology কী

Persistent homology এমন একটি পদ্ধতি যা দেখে:
- কোন connected component কখন জন্ম নিল
- কখন merge হয়ে গেল
- কোন loop কখন তৈরি হলো
- কখন ভেঙে গেল

### 8.3 H0 এবং H1

- **H0**: connected components
- **H1**: loops বা 1-dimensional holes

খুব সহজভাবে:
- H0 বলে data কতগুলো আলাদা cluster-এর মতো আছে
- H1 বলে data-এর মধ্যে ring/loop-এর মতো structure আছে কি না

### 8.4 Persistence diagram

Persistence diagram হলো অনেকগুলো pair-এর collection:
- birth time
- death time

প্রতিটি pair বলে একটি topological feature কখন শুরু আর কখন শেষ হয়েছে।

যদি feature দীর্ঘ সময় থাকে, তাহলে সেটাকে important structure ধরা হয়।

## 9. Vietoris-Rips filtration

Persistence diagram বের করতে PRISM point cloud-এর ওপর **Vietoris-Rips filtration** ব্যবহার করে।

এটা কী:
- প্রথমে point cloud-এর প্রতিটি point আলাদা
- radius বাড়ালে কাছের points connect হতে শুরু করে
- connect হয়ে components, triangles, loops তৈরি হয়
- radius আরও বাড়লে structure বদলায়

এই পরিবর্তনের timeline থেকেই persistent homology পাওয়া যায়।

## 10. Wasserstein distance কী

Wasserstein distance সাধারণভাবে দুই distribution বা two point sets-এর মধ্যে transportation cost বোঝায়।

PRISM-এ এটি দুই persistence diagram-এর দূরত্ব হিসেবে ব্যবহৃত হয়।

Diagram-এ থাকা points-গুলোকে one-to-one match করতে হয়, কিছু points diagonal-এ project হতে পারে, এবং total matching cost minimize করা হয়।

সহজ ভাষায়:
- clean input-এর topology-এর সঙ্গে current input-এর topology কতটা আলাদা
- আলাদা হলে distance বড় হবে
- distance বড় মানে anomaly score বেশি

## 11. TAMM: Topological Activation Manifold Monitor

TAMM হলো PRISM-এর core detector।

### Process

1. clean calibration set থেকে sample নেওয়া হয়
2. প্রতিটি sample-এর monitored layer activation বের করা হয়
3. activation থেকে point cloud বানানো হয়
4. persistence diagram তৈরি করা হয়
5. প্রতিটি layer-এর জন্য একটি clean reference diagram বানানো হয়
6. inference time-এ নতুন input এলে তার diagram-এর সঙ্গে reference diagram তুলনা করা হয়
7. Wasserstein distance দিয়ে score বের হয়
8. layer-wise score average করে final anomaly score পাওয়া যায়

### Formula

Layer-wise score:
$$
s_\ell(x) = W_2(\mathrm{Dgm}(\phi_\ell(x)), \mathcal R_\ell)
$$

Overall score:
$$
S(x) = \frac{1}{L}\sum_{\ell=1}^{L} s_\ell(x)
$$

এখানে:
- $x$ = input image
- $\phi_\ell(x)$ = layer $\ell$-এর activation
- $\mathrm{Dgm}(\cdot)$ = persistence diagram
- $\mathcal R_\ell$ = layer $\ell$-এর reference diagram
- $W_2$ = 2-Wasserstein distance
- $L$ = total monitored layers

### Intuition

- clean image হলে score সাধারণত ছোট
- adversarial হলে score বড় হওয়ার সম্ভাবনা বেশি
- score যত বড়, activation topology clean reference থেকে তত দূরে

## 12. Medoid কী

PRISM clean reference হিসেবে **Wasserstein medoid** ব্যবহার করে।

Medoid হলো dataset-এর এমন একটি actual sample, যার distance অন্য সব sample-এর কাছে মোটের ওপর সবচেয়ে কম।

মানে average representative-এর মতো।

পেপারে reference diagram এমন diagram যা clean calibration diagrams-এর মধ্যে সবচেয়ে representative।

## 13. CADG: Conformal Adversarial Detection Guarantee

এটা PRISM-এর formal guarantee অংশ।

### Conformal prediction কী

Conformal prediction হলো এমন একটি method যা calibration data থেকে threshold বসিয়ে finite-sample guarantee দিতে পারে।

এখানে লক্ষ্য prediction label না, বরং anomaly score threshold।

### কীভাবে কাজ করে

- clean calibration scores সংগ্রহ করা হয়
- তাদের order করা হয়
- desired false positive rate $\alpha$ অনুযায়ী quantile বেছে নেওয়া হয়

Formula:
$$
\hat q_\alpha = s_{(\lceil (n+1)(1-\alpha) \rceil)}
$$

এখানে $s_{(k)}$ হলো $k$-th order statistic।

### Guarantee

যদি test input calibration distribution-এর মতো clean distribution থেকে আসে, তাহলে:
$$
\Pr[S(x_{test}) > \hat q_\alpha] \le \alpha + \frac{1}{n+1}
$$

এর মানে clean input ভুলভাবে flag হওয়ার rate bounded থাকে।

### তিনটি tier

PRISM তিনটি severity tier ব্যবহার করে:
- L1: mild warning
- L2: stronger suspicion
- L3: highest severity / reject or expert routing

Validation artifact অনুযায়ী clean gate:
- L1 FPR = 0.066
- L2 FPR = 0.015
- L3 FPR = 0.002

সবগুলো target-এর ভেতরে:
- L1 <= 0.10
- L2 <= 0.03
- L3 <= 0.005

## 14. FPR কী

**FPR** = False Positive Rate.

অর্থ:
- clean sample কতবার ভুল করে adversarial/suspicious হিসেবে flag হলো

Formula:
$$
FPR = \frac{FP}{FP + TN}
$$

যেখানে:
- FP = false positive
- TN = true negative

PRISM-এর জন্য FPR খুব গুরুত্বপূর্ণ, কারণ detector বেশি sensitive হলে clean sample ভুলে attack হিসেবে ধরে user experience নষ্ট হতে পারে।

## 15. TPR কী

**TPR** = True Positive Rate.

অর্থ:
- actual adversarial sample-এর কত অংশ correctly detected হলো

Formula:
$$
TPR = \frac{TP}{TP + FN}
$$

যেখানে:
- TP = true positive
- FN = false negative

TPR বেশি হলে detector attack ধরতে ভালো।

## 16. SACD: Sequential Adversarial Campaign Detection

এটা PRISM-এর temporal monitoring অংশ।

### কেন দরকার

Attack এক-একটা isolated query দিয়ে নাও আসতে পারে। Attackers আগে harmless-looking probe পাঠিয়ে system বুঝে নিতে পারে, তারপর coordinated campaign চালাতে পারে।

### কীভাবে কাজ করে

PRISM score stream observe করে:
$$
S(x_1), S(x_2), S(x_3), \dots
$$

এখানে Bayesian Online Changepoint Detection, বা BOCPD, ব্যবহার করা হয়।

BOCPD score stream-এ distribution shift detect করে।

Key idea:
- clean phase-এ scores stable থাকে
- campaign শুরু হলে score distribution shift করে
- changepoint probability বাড়ে

### Trigger condition

$$
\Pr[r_t \le k_{alert}] > p_{alert}
$$

এখানে:
- $r_t$ = last changepoint থেকে elapsed run length
- $k_{alert}$ = recent change কতটা recent হলে alert ধরবে
- $p_{alert}$ = probability threshold

Paper-এর default values:
- $k_{alert}=5$
- $p_{alert}=0.50$
- warmup period = 20

### Interpretation

যদি score stream আচমকা বেড়ে যায়, detector campaign mode চালু করে, আর threshold আরও sensitive করে।

## 17. L0 mode কী

Paper-এ campaign detector active হলে threshold multiplier $\lambda$ ব্যবহার করা হয়।

Default:
- $\lambda = 0.8$

মানে threshold 20% কমিয়ে দেওয়া হয়:
- detection বেশি sensitive হয়
- borderline adversarial ধরার সম্ভাবনা বাড়ে
- কিন্তু FPR কিছুটা বাড়তে পারে

Ablation-এ দেখা যায় L0 mode TPR বাড়ায়, কিন্তু FPR cost আছে।

## 18. TAMSH: Topology-Aware MoE Self-Healing

L3 tier-এর input PRISM hard reject না করে expert subnet-এ পাঠাতে পারে।

### কেন

যদি input খুব সন্দেহজনক হয়, তখন rejection service availability কমিয়ে দিতে পারে।
Paper বলে, adversarial input-ও prediction service থেকে empty answer না দিয়ে topology-matched expert দিয়ে rescue করা যেতে পারে।

### MoE কী

MoE = Mixture of Experts

মানে multiple expert subnet আছে, এবং input-এর structure অনুযায়ী appropriate expert বেছে নেওয়া হয়।

### Routing formula

$$
k^*(x) = \arg\min_{k=1}^{K} W_2\bigl(\mathrm{Dgm}(\phi_L(x)), \mathcal R_k^L\bigr)
$$

এখানে:
- $K$ = expert সংখ্যা
- $\mathcal R_k^L$ = expert cluster-এর medoid/reference at final layer
- nearest topology-matching expert বেছে নেওয়া হয়

### Result

Paper claim করে L3-triggered input-এ expert routing prediction availability 100% রাখে।

## 19. Attack types explained

### 19.1 FGSM

**FGSM** = Fast Gradient Sign Method.

এটা single-step gradient attack।

Idea:
- loss-এর gradient বের করো
- gradient-এর sign নাও
- input-এ ছোট perturbation যোগ করো

Formula-ভাবনা:
$$
\tilde x = x + \epsilon \cdot \mathrm{sign}(\nabla_x \mathcal L)
$$

এখানে $\epsilon$ perturbation budget.

Meaning:
- খুব দ্রুত attack
- কিন্তু iterative attack-এর তুলনায় weaker হতে পারে
- PRISM-এর জন্য FGSM hardest low-end case, কারণ subtle topology shift তৈরি করতে পারে

### 19.2 PGD

**PGD** = Projected Gradient Descent.

এটা FGSM-এর stronger version বলা যায়।

Idea:
- small gradient step repeatedly নাও
- প্রতিবার perturbation budget-এর মধ্যে project করে রাখো
- বহু step শেষে strong adversarial example তৈরি হয়

Paper-এ PGD-40 ব্যবহার করা হয়েছে:
- 40 step
- $\epsilon = 0.03$ বা 8/255 convention-এ
- step size = $\epsilon/4$

PGD সাধারণত strong white-box attack।

### 19.3 CW / C&W

**CW** = Carlini & Wagner attack.

Paper-এ CW-L2 উল্লেখ আছে।

এটা optimization-based attack।

Goal:
- classifier fool করা
- perturbation-এর $\ell_2$ norm ছোট রাখা

Meaning of $\ell_2$:
$$
\|\delta\|_2 = \sqrt{\sum_i \delta_i^2}
$$

CW attack সাধারণত subtle কিন্তু powerful।

Paper-এ CW-L2 as a cited attack আছে, যদিও main retained local artifact-এ FGSM, PGD, Square-ই verified হয়েছে।

### 19.4 Square Attack

Square Attack হলো black-box score-based attack.

Idea:
- gradient জানা লাগে না
- random square patches modify করে query-based search করা হয়
- model output দেখে adversarial direction refine করা হয়

Paper-এ এটা 5000 query-তে চালানো হয়েছে।

### 19.5 AutoAttack

**AutoAttack** কোনো single attack না; এটা standardized strong attack suite.

এতে সাধারণত multiple robust attacks combine করা হয় যাতে evaluation reliable হয়।

সহজভাবে:
- একটা model robust কিনা যাচাই করতে AutoAttack benchmark হিসেবে ব্যবহার করা হয়
- এই suite সাধারণত FGSM/PGD/CW-এর মতো ideas-এর mix বা variants অন্তর্ভুক্ত করে

PRISM manuscript-এ AutoAttack main experiment attack হিসেবে used হয়নি, কিন্তু robust evaluation context-এ term হিসেবে relevant।

## 20. White-box, black-box, query-based

### White-box attack

Attacker model-এর ভিতরের details জানে:
- gradients
- architecture
- parameters

FGSM, PGD, CW সাধারণত white-box context-এ পড়ে।

### Black-box attack

Attacker gradient জানে না।

Model-কে query করে result দেখে attack চালায়।

Square Attack black-box ধরনের।

### Query-based

Attack কত query নিয়েছে, তা important metric হতে পারে।

Square Attack-এর মতো attacks query count ব্যবহার করে optimize করে।

## 21. Norms: $\ell_\infty$ এবং $\ell_2$

### $\ell_\infty$

এটা per-pixel সর্বোচ্চ change সীমা।

যদি $\epsilon=8/255$, তাহলে প্রতিটি pixel-এর change এর upper bound প্রায় 0.03137 normalized scale-এ।

FGSM, PGD, Square-এ এই budget ব্যবহার করা হয়েছে।

### $\ell_2$

এটা পুরো perturbation vector-এর overall size।

CW-L2-তে perturbation এমনভাবে minimize করা হয় যাতে total Euclidean shift ছোট থাকে।

## 22. Threshold tier system

PRISM-এ scores তিন স্তরে ভাগ করা হয়:

- **L1**: low suspicion
- **L2**: moderate suspicion
- **L3**: high suspicion / reject / expert route

Verified validation gate অনুযায়ী:
- L1 threshold control quite permissive
- L2 tighter
- L3 strictest

Retention artifact-এ clean distribution:
- PASS = 469
- L1 = 22
- L2 = 8
- L3_REJECT = 1

মানে বেশিরভাগ clean input pass করেছে, অল্প কিছু top tier-এ গেছে।

## 23. Verified numbers from retained artifact

### 23.1 Single-seed retained evaluation

Artifact: `results_n500_optimized_20260421.json`

Main numbers:
- FGSM TPR = 0.844
- PGD TPR = 1.000
- Square TPR = 0.924
- FPR = 0.062 for all three attacks
- Clean samples = 500
- Adv samples = 500 per attack
- Latency mean = 73.68 ms
- Latency p95 = 101.13 ms

Confusion counts:
- FGSM: TP 422, FP 31, FN 78, TN 469
- PGD: TP 500, FP 31, FN 0, TN 469
- Square: TP 462, FP 31, FN 38, TN 469

Tier FPR:
- L1+ = 0.062
- L2+ = 0.018
- L3+ = 0.002

### 23.2 Validation gate

Artifact: `ensemble_fpr_report.json`

On 1000 clean validation images:
- L1 FPR = 0.066
- L2 FPR = 0.015
- L3 FPR = 0.002

সবগুলো target pass করেছে।

### 23.3 Baselines

Baseline detector artifact shows:
- LID: FGSM 0.998 TPR, FPR 0.088
- Mahalanobis: FGSM 1.000 TPR, FPR 0.104
- PGD সব ক্ষেত্রে প্রায় 1.0 TPR
- Square-এও strong performance

PRISM-এর advantage হলো formal conformal FPR control + layered response, not just raw detection.

## 24. manuscript numbers vs verified artifact

Manuscript-এর sections-এ কিছু older or narrative numbers আছে:
- abstract-এ PGD 100%, Square 94%, FGSM approx 50%, FPR 8%
- experiments section-এ FGSM 62%, PGD 100%, Square 53%, FPR 1.6%
- conclusion-এ PGD 100%, Square 94%, FGSM approx 50%, FPR 8%

Verified local artifact-এ আবার অন্য set পাওয়া গেছে:
- FGSM 84.4%
- PGD 100%
- Square 92.4%
- FPR 6.2%

এ কারণে reporting-এর সময় source consistency খুব গুরুত্বপূর্ণ। Artifact-derived numbers-ই বর্তমান local reference হিসেবে বেশি reliable।

## 25. PRISM কীভাবে clean data ব্যবহার করে

একটি subtle but important point হলো threshold fitting-এর জন্য adversarial data ব্যবহার করা হয় না।

Process:
- profile/calibration split clean থাকে
- calibration clean data দিয়ে threshold fit হয়
- validation clean data দিয়ে FPR verify করা হয়
- test/eval data-তে final detection চালানো হয়

এটা data split hygiene-এর জন্য গুরুত্বপূর্ণ।

## 26. split conformal guarantee কেন important

অনেক detector শুধু empirical accuracy report করে।
PRISM তার সঙ্গে statistical guarantee দিতে চায়।

মানে:
- যদি calibration ও test একই clean distribution থেকে আসে
- তাহলে clean image ভুল flag হওয়ার probability control করা যায়

এটা practical deployment-এ বেশি গুরুত্বপূর্ণ, কারণ false alarm rate fixed না থাকলে system noisy হয়ে যায়।

## 27. campaign detection কী বোঝায়

Attackers অনেক সময় একবারে final attack দেয় না।

তারা আগে small query sequence দিয়ে model-এর weak point দেখে নেয়।

PRISM-এর SACD এই kind of sequential behavior monitor করে:
- clean stream stable
- attack campaign শুরু হলে score shift
- BOCPD সেটা ধরতে পারে

এভাবে PRISM শুধু per-image নয়, per-session defense-ও দেয়।

## 28. self-healing কেন দরকার

অনেক detector suspicious input detect করে reject করে।

Reject করলে security বাড়ে, কিন্তু availability কমে।

PRISM বলে:
- very high severity হলে expert subnet দিয়ে prediction rescue করা যায়
- ফলে system fully unavailable না হয়

এটা especially service-oriented deployment-এ useful।

## 29. limitations

Paper নিজেই কিছু limitation মেনে নেয়:
- persistent homology computationally heavy
- high-dimensional activations subsample করতে হয়
- CPU latency বেশি হতে পারে
- adaptive attacks score function directly target করতে পারে
- distribution shift FPR বাড়াতে পারে

Verified artifacts অনুযায়ী latency এখনো moderate-to-high bottleneck।

## 30. ভবিষ্যৎ কাজ

Paper যে future directions দেয়:
- transformer-এ extend করা
- adaptive topology-aware attacks defend করা
- randomized smoothing-এর সঙ্গে combine করা
- online recalibration করা
- distribution shift handle করা

## 31. এক লাইনে পুরো PRISM

PRISM হলো একটি runtime adversarial defense system যা network-এর ভিতরের activation topology দেখে suspicious input detect করে, clean calibration দিয়ে threshold certificate দেয়, campaign-level attack ধরতে পারে, আর high-severity case-এ expert subnet দিয়ে recovery করার চেষ্টা করে।

## 32. beginner-friendly summary

যদি খুব সহজে বলি:
- image network-এ ঢোকে
- network-এর ভিতরের feature shape দেখি
- clean image-এর shape-এর সাথে compare করি
- shape খুব আলাদা হলে suspicious ধরি
- clean data দিয়ে safe threshold বসাই
- continuous attack হলে alert mode চালু করি
- খুব dangerous input হলে alternative expert দিয়ে predict করি

এটাই PRISM।

## 33. short glossary

- **Attack**: model-কে ভুল করানোর input perturbation
- **Adversarial example**: attack করা input
- **Clean input**: normal, unmodified input
- **Calibration set**: threshold fit করার clean data
- **Validation set**: threshold verify করার data
- **Threshold**: score-এর cutoff
- **Score**: suspiciousness measure
- **Reference profile**: clean topology summary
- **Medoid**: representative sample
- **Persistence diagram**: topology summary output
- **Wasserstein**: diagram distance metric
- **Conformal prediction**: statistical thresholding framework
- **BOCPD**: sudden distribution shift detector
- **MoE**: multiple expert subnet system

## 34. শেষ কথা

PRISM-এর main novelty তিন জায়গায়:
- ভিতরের activation topology ব্যবহার করে anomaly scoring
- conformal prediction দিয়ে FPR control
- campaign detection + self-healing routing যোগ করা

কিন্তু exact paper claim আর artifact result একই না-ও হতে পারে, তাই analysis করার সময় source-কে আলাদা করে দেখা জরুরি।

---

### Referenced verified files
- [prism/experiments/evaluation/results_n500_optimized_20260421.json](prism/experiments/evaluation/results_n500_optimized_20260421.json)
- [prism/experiments/calibration/ensemble_fpr_report.json](prism/experiments/calibration/ensemble_fpr_report.json)
- [prism/experiments/evaluation/run_report_n500_local_20260420.md](prism/experiments/evaluation/run_report_n500_local_20260420.md)
- [prism/paper/main.tex](prism/paper/main.tex)
- [prism/paper/sections/method.tex](prism/paper/sections/method.tex)
- [prism/paper/sections/experiments.tex](prism/paper/sections/experiments.tex)
- [prism/paper/sections/conclusion.tex](prism/paper/sections/conclusion.tex)
