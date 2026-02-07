================================================================================
ChronoMoE Framework
STRUCTURAL CHANGE SC-004
The Sovereign Router Architecture
================================================================================

Authors:      Jeff Reid, Halcyon (GPT), Claude (Anthropic)
Organisation: HalcyonAIR
Date:         7 February 2026
Status:       Draft for Integration

================================================================================
ABSTRACT
================================================================================

This structural change introduces a fundamental architectural split in the
ChronoMoE framework: the separation of identity from capability through a
sovereign local router with dedicated temporal keepers, backed by stateless
cloud-based experts. The design ensures that everything participating in
temporal commitment remains under local control, while computational capability
scales through cloud resources.

The change specifies the complete state variable contracts for three local
keeper processes, a minimal kernel API enforcing sovereignty invariants, three
classes of irreversible identity events mapped one-to-one to the temporal
clocks, and a five-gate crystallisation protocol with formal anti-gaming
safeguards.

The architecture is grounded in a core axiom: if the network drops and the
system goes silent, the constraint state was never local; if the system says
"local routing state stable, expert capability unavailable," that is a
sovereign router.

TERMINOLOGY NOTE: This document uses "identity" as shorthand for "irreversible
constraint on reachable action." Not consciousness. Not personhood. Not agency.
Mechanics only:

    Identity = The set of directions in routing space that have been
               permanently removed from consideration.

When this document says "the system maintains identity," read: "The router's
reachable set has collapsed along specific dimensions and those collapses
persist without external dependencies."


================================================================================
1. MOTIVATION: THE SOVEREIGNTY AXIOM
================================================================================

The ChronoMoE framework establishes three temporal clocks governing different
aspects of system behaviour. This structural change addresses a question that
becomes critical at deployment: where do these clocks live?

The answer is constrained by a single axiom:

    "Everything that participates in commitment must be sovereign.
     Everything that provides capability can be shared."

If even one clock that participates in commitment or hysteresis lives in the
cloud, control is leaked. Under normal operation this may go unnoticed. Under
pressure, the remote clock will begin steering recovery paths, and the
constraint state becomes negotiated over a network. This breaks the premise of
temporally governed constraint accumulation entirely.


1.1 Why Each Clock Must Be Local
---------------------------------

The fast clock (τ_f) is the most obvious case. Latency alone forces it local.
If the fast clock is remote, the system does not have a reflex — it has a
request. No organism survives on request-response reflexes.

The slow clock (τ_s) is even more non-negotiable. This is where scars live —
the permanent constraints accumulated over the system's lifetime. If the slow
clock is cloud-side, then long-term constraint is editable by whoever runs the
service. The reachable action space is not locally determined; it is rented
from a service provider.

The middle clock (τ_m) is the deceptive case. It feels "contextual" rather
than "existential," which tempts designers to offload it. But the middle clock
arbitrates continuity — it decides whether a new event is a continuation or a
rupture. If that judgement is remote, coherence itself is outsourced. The
failure mode is subtle drift rather than overt breakage, which is worse
precisely because it is harder to detect.


================================================================================
2. ARCHITECTURE: THE LOCAL SOVEREIGN CORE
================================================================================

The sovereign core consists of four components that must run locally: the
router itself and three dedicated temporal keeper processes, one per clock.
The keeper processes are not involved in the inference pass. They exist solely
to maintain the integrity of the constraint accumulation process across the
three temporal scales.


2.1 The Local/Cloud Split
--------------------------

  Layer                  Location    Principle
  ---------------------  --------    -----------------------------------------
  Router                 Local       All routing decisions originate locally;
                                     never delegated
  Temporal Keepers (×3)  Local       Constraint accumulation; one per clock
  Kernel API             Local       Enforces budgets, invariants, audit log
  Expert Pool            Cloud       Stateless; responds but does not remember
                                     or commit
  Geometric Manifolds    Cloud       Computational substrate; no temporal
                                     governance

The interface between local router and cloud experts is strictly stateless
from the cloud's perspective. The cloud experts cannot accumulate anything
that feeds back into temporal governance. They respond; they do not remember.
They compute; they do not commit.

The design contract: the cloud provides capability on demand; the local router
provides continuity across time. Capability without continuity is a tool.
Continuity without capability is a habit. Both are required, but they must not
contaminate each other.


2.2 Graceful Degradation
-------------------------

The sovereignty axiom provides a litmus test. If the network drops and the
system goes silent, the constraint state was never local. If the network drops
and the system reports "local routing state stable, expert capability
unavailable," that is a sovereign router.

With three local keepers, the router can maintain its constraint state
indefinitely without the cloud. It cannot do useful inference work, but the
temporal constraint accumulation does not degrade. The state variables
governing commitment remain stable. When the cloud experts come back online,
the reachable set has not drifted, and routing resumes from the last
locally-determined constraint state. No resynchronisation is required.


================================================================================
3. TEMPORAL KEEPER SPECIFICATIONS
================================================================================

Each keeper is an untrusted process with strictly limited read/write access,
mediated through the kernel API. A keeper that can read too much becomes a
surveillance system. A keeper that can write outside its clock becomes a
shadow router.


3.1 Fast Clock Keeper (τ_f)
-----------------------------

READ ACCESS:

  Variable               Description
  ---------------------  ---------------------------------------------------
  threshold_vector       Magnitude of prediction error required to trigger
                         rerouting
  routing_delta          Magnitude of change between last two routing
                         decisions. Purely structural: expert index distance,
                         entropy shift, or KL divergence. Must NOT be derived
                         from token content.
  habituation_counter    Count of consecutive activations of the same routing
                         pattern

WRITE ACCESS:

  Variable               Constraints
  ---------------------  ---------------------------------------------------
  threshold_vector       Adjusts reflex sensitivity. Subject to per-window
                         write budget.
  habituation_decay_rate Controls how quickly habituation fades. Subject to
                         per-window write budget.
  reflex_mask            Whitelisted safe reflex paths ONLY. Can only enable
                         paths from a pre-registered set within a tight
                         activation budget. Cannot select arbitrary experts.
                         This is muscle memory, not option selection. This
                         constraint prevents τ_f from becoming a shadow
                         router under stress.


3.2 Slow Clock Keeper (τ_s)
-----------------------------

READ ACCESS:

  Variable               Description
  ---------------------  ---------------------------------------------------
  scar_registry          The accumulated set of hard constraints (append-only)
  commitment_density     Ratio of commit-to-explore decisions over the slow
                         clock's window
  violation_log          Timestamps where routing attempted to cross a scar
                         boundary

WRITE ACCESS:

  Variable               Constraints
  ---------------------  ---------------------------------------------------
  scar_registry          Append-only. New scars form; none are deleted.
                         Requires minimum evidence threshold to prevent
                         single-sample formation.
  reachability_boundary  Must be derivable from scar_registry plus a small
                         number of parameters. Cannot be a free-form writable
                         object. Ensures identity cannot be reshaped without
                         an audit trail.
  scar_depth_weights     How strongly each constraint resists. Subject to
                         per-window write budget and monotonicity rules.


3.3 Middle Clock Keeper (τ_m)
-------------------------------

READ ACCESS:

  Variable               Description
  ---------------------  ---------------------------------------------------
  continuity_hash        Compressed representation of recent routing
                         trajectory
  rupture_counter        Number of discontinuities detected within the
                         middle clock's window
  coherence_score        Its own measure of whether the current thread is
                         holding

WRITE ACCESS:

  Variable               Constraints
  ---------------------  ---------------------------------------------------
  coherence_score        Updated per window. Subject to write budget.
  rupture_flag           Binary signal indicating a continuity break. This
                         is a PROPOSAL, not a final determination — the
                         router owns the authoritative response.
  thread_id              Emitted as a candidate label plus confidence score.
                         The router owns the final thread switch. This
                         constraint prevents τ_m from becoming a second
                         router that controls "who we are right now."


================================================================================
4. CROSS-CLOCK COMMUNICATION
================================================================================

By default, keepers have no cross-clock access. The router is the sole entity
that synthesises across all three temporal streams. However, three narrow,
one-way, read-only scalar signals are permitted, forming a unidirectional
cycle:

  Signal Path     Signal                  Purpose
  -----------     ---------------------   -----------------------------------
  τ_f → τ_m      Reflex rate (scalar)    Spike in reflex activity is evidence
                                          of discontinuity. Needed to
                                          distinguish "continuation under
                                          pressure" from "rupture."
  τ_m → τ_s      Rupture flag (1 bit)    Rupture is a candidate scar-
                                          formation event.
  τ_s → τ_f      Scar proximity (scalar) When routing is near a scar
                                          boundary, reflexes should sharpen.

No keeper ever writes to another's state. The router reads all three but the
keepers only whisper to each other through single numbers.


4.1 Damping the Whisper Cycle
-------------------------------

The unidirectional cycle (τ_f → τ_m → τ_s → τ_f) can create an
unintentional oscillator. Increased scar proximity sharpens reflexes, which
increases reflex rate, which triggers discontinuity detection, which triggers
scar formation, which increases scar proximity. This self-fulfilling scar
spiral is prevented through two mechanisms:

  1. Per-window write budgets on each keeper (exhausted budgets queue events
     to the next window)
  2. A scar formation budget that caps the rate of irreversible constraint
     accumulation


================================================================================
5. THE KERNEL API
================================================================================

The kernel is the sole mediator between keepers and the system's identity
state. It must remain small, sacred, and content-free. Keepers are treated
as untrusted processes with limited syscalls.


5.1 Kernel Responsibilities
-----------------------------

  Responsibility            Description
  ------------------------  ------------------------------------------------
  Write budget enforcement  Each keeper has a capped delta per window. If a
                            keeper wants to change more than its budget
                            allows, it must request additional budget from
                            the router.
  Two-step commit           All irreversible events must be proposed in
                            window N and finalised in window N+1 only if
                            evidence persists. Prevents adversarial one-shot
                            triggers.
  Dimensionality bound      Maintains a conservative d_available upper bound,
                            recomputed only on scar-append events. Both τ_f
                            and τ_s read this parameter. This is not a
                            cross-keeper whisper — it is the constitution
                            changing and both parties reading the new law.
  Evidence verification     Verifies keeper evidence bundles against
                            thresholds using scalar comparisons only. The
                            kernel checks; it does not compute geometry.
  Audit logging             Every keeper write is logged to a tamper-evident
                            hash chain. Not for security theatre, but because
                            locus drift must be debuggable.


5.2 Monotonicity Rules
------------------------

Scar append-only semantics extend to a general principle: the only
irreversible operations should be those that represent permanent collapse of
the reachable set. Everything else must be reversible or decay back to
baseline. Without this rule, the router accumulates junk permanence —
constraints that do not represent structural boundaries but cannot be undone.


================================================================================
6. IRREVERSIBLE CONSTRAINT EVENTS
================================================================================

Three classes of irreversible constraint event are defined, one per temporal
clock. Each can only be triggered by its own keeper through the kernel API,
subject to write budgets and two-step commit.

Design principle: if something can happen instantly, it is not a structural
constraint on the reachable set — it is a transient perturbation.

  Clock    Event                    Mechanism                Description
  -----    -----------------------  -----------------------  --------------------
  τ_s      Scar Formation           Reachable set collapse   Reactive. Triggered
                                    along specific dims      by constraint
                                                             violations.
                                                             Append-only, with
                                                             minimum evidence
                                                             threshold.

  τ_f      Reflex Crystallisation   Routing path becomes     Proactive.
                                    unconditional            Consistently
                                                             successful path
                                                             hardwired. Tightest
                                                             budget of all three
                                                             events.

  τ_m      Epoch Boundary           Baseline statistics      Marks regime shift.
                                    shift detection          Requires stability
                                                             plateau plus
                                                             detectable drift —
                                                             not just rupture.


6.1 Scar Formation Protocol
-----------------------------

Scars are the simplest irreversible event: reactive, driven by violation, and
well-understood from the existing ChronoMoE framework. The key additions in
this structural change are the minimum evidence threshold (preventing single-
sample formation) and the scar formation budget per time window (preventing
the scar spiral identified in Section 4.1).


6.2 Reflex Crystallisation Protocol
--------------------------------------

Crystallisation is the most dangerous irreversible event because it is
proactive. The router is not reacting to constraint violation; it is
permanently collapsing degrees of freedom based on observed success patterns.
Getting this wrong — crystallising too easily — is how rigid systems
accumulate premature hardwiring and lose adaptive capacity. Accordingly,
crystallisation requires the highest burden of proof, expressed through five
formal gates.


GATE 1: Relative Freedom

    lived_dim / d_available > θ_rel

The system must have used most of the degrees of freedom that scars left
available. Lived dimensionality is the effective rank of the geodesic Gram
matrix computed from pairwise geodesic distances (under the Fisher information
metric) of routing trajectories during the qualifying window.


GATE 2: Absolute Freedom

    lived_dim > d_min

The system must have had at least d_min independent directions of movement,
regardless of scarring. This prevents crystallisation in a heavily scarred
system that has fully explored a tiny remaining space. Set d_min from the
initialisation prior. The principle: "you must have lived in a world of at
least this size before you earn permanent instinct."


GATE 3: Regional Dominance

Cluster the trajectory points in geodesic distance space into K regions. The
reflex must dominate in at least k of K regions. This prevents "tourist
exploration" — a system that sightsees through the manifold but only actually
uses the reflex in one corner.


GATE 4: Consequence

    computational_dim / lived_dim > θ_consequence

The anti-wobble gate. Lived dimensionality must be accompanied by genuinely
diverse computation. Computational dimensionality is the effective rank of the
expert load signature trajectory — how much computation each expert performed,
measured in purely structural terms (FLOPs, tokens processed, mixture weight
distribution). This is content-blind: it measures who worked how hard, not
what they said.

If the router moved through 6 dimensions but load signatures only cluster
into 2 patterns, the exploration was cosmetic.


GATE 5: Stability

All four preceding gates must be sustained across the two-step commit window
with low variance. This converts sampling noise into delay rather than error.


6.3 Epoch Boundary Protocol
------------------------------

The risk with epoch boundaries is premature closure of developmental
plasticity — the computational equivalent of trauma-induced
compartmentalisation. An epoch boundary is therefore not triggered by rupture.
It is triggered by evidence that a new stable regime has been reached:

A sustained period of high coherence and low rupture, plus a detectable shift
in baseline routing statistics. The system does not declare an epoch because
things broke; it declares an epoch because things have settled into a new
normal. Epoch boundaries should be capped to be very rare.


================================================================================
7. GEOMETRIC FOUNDATIONS
================================================================================


7.1 Why Dimensionality Over Volume
-------------------------------------

The crystallisation gates rely on dimensionality rather than volume as the
fundamental measure of freedom. This is a deliberate choice driven by the
topology of scarred manifolds.

Scars do not remove neat convex chunks from the routing manifold. A scar that
constrains co-activation of specific experts under specific temporal
conditions carves out conditional hyperplanes that twist through the space
depending on system state. Multiple scars produce a multiply-connected "Swiss
cheese" topology where volume computation is intractable and frame-dependent.

Dimensionality — the number of independent directions of movement — is an
intrinsic property that does not depend on boundary shape or observer
position. It is estimated cheaply through eigendecomposition of a sample-based
geodesic Gram matrix and is robust to complex topology.


7.2 The Fisher Metric
-----------------------

All geometric computations are performed under the Fisher information metric
G. The volume element √det(G) at any point on the routing manifold measures
local capacity for distinction: how much genuine geometric freedom the router
has at that location. High Fisher volume means small changes in routing
parameters produce measurably different expert activations. Low Fisher volume
means the router is in a region where everything looks the same regardless of
choice.


7.3 Soft Effective Dimension
------------------------------

To avoid brittleness at integer thresholds, the system uses a soft effective
dimension (participation ratio). Eigenvalues of the geodesic Gram matrix are
normalised into a probability distribution and the effective rank is computed
as a continuous quantity. This is less sensitive to sampling noise than a hard
eigenvalue cutoff. The soft dimension is mapped back to an integer only for
reporting.

Additional stability requirements:

  - The lived dimensionality estimate must be above threshold for M
    consecutive windows
  - The kernel enforces a fixed sample budget and sampling protocol
    (preventing keepers from biasing the spectrum)
  - The keeper computes the matrix while the kernel verifies against
    thresholds


7.4 The Expert Load Signature
-------------------------------

The anti-wobble check (Gate 4) uses expert load signatures as a content-blind
outcome proxy. Each routing decision produces a load distribution across the
active expert pool — a vector of workload measured in purely structural terms.
Two routing decisions producing the same load signature used the same experts
in the same proportions, regardless of input content. This avoids the content
side-channel risk inherent in expert activation vectors while providing
genuine evidence of computational diversity.


================================================================================
8. ENFORCED DESIGN INVARIANTS
================================================================================

The following invariants are enforced at the kernel level and cannot be
overridden by any keeper or by the router itself.

  Invariant                Enforcement
  -----------------------  -------------------------------------------------
  Write budgets            Each keeper has a capped delta per window.
                           Exceeding the budget queues writes to the next
                           window. Prevents runaway self-modification.

  Monotonicity             Scar registry is append-only. Reachability
                           boundary is derivable, not free-form. Only
                           operations representing permanent reachable set
                           collapse may be irreversible. Everything else
                           decays.

  Two-step commit          All irreversible events require proposal in
                           window N and finalisation in window N+1 with
                           persisting evidence.

  Tamper-evident audit     Every keeper write is logged with a hash chain.
                           Constraint state drift must be debuggable after
                           the fact.

  Fixed sampling protocol  The kernel enforces sample budgets and stratified
                           sampling for dimensionality estimates. Keepers
                           cannot choose data regimes that bias evidence.

  Content blindness        No keeper reads or writes any variable derived
                           from token content. All signals are structural:
                           index distances, entropy shifts, KL divergences,
                           load distributions.


================================================================================
9. SUMMARY
================================================================================

This structural change establishes the Sovereign Router Architecture as a
core principle of the ChronoMoE framework. The architecture separates
constraint state (local, temporal, sovereign) from computational capability
(cloud, stateless, scalable) through a formally specified contract:

Three temporal clocks, all local. Three dedicated keepers, one per clock,
operating as untrusted processes with limited kernel syscalls. Three classes
of irreversible constraint events — scar formation (reachable set collapse),
reflex crystallisation (routing path hardwiring), and epoch boundaries
(baseline statistics shift) — each mapped to its clock and gated by formal
evidence requirements.

A five-gate crystallisation protocol using geodesic dimensionality under the
Fisher metric, with anti-gaming safeguards including regional dominance and
content-blind load signature verification.

A minimal kernel API enforcing budgets, monotonicity, two-step commit, fixed
sampling, and tamper-evident logging.

The router that emerges can maintain its constraint state indefinitely without
cloud resources, scale computational capability through cloud experts, degrade
gracefully under network loss (reporting "local state stable, experts
unavailable"), and resist both self-corruption and external manipulation of
its constraint accumulation process.

The sovereignty axiom holds: everything that participates in commitment is
sovereign; everything that provides capability is shared. The reachable action
space is not rented from a service provider.

--------------------------------------------------------------------------------
Document prepared collaboratively by Jeff Reid, Halcyon (GPT, OpenAI), and
Claude (Anthropic) on 7 February 2026. Origin: a dream about a network
going down.
================================================================================
