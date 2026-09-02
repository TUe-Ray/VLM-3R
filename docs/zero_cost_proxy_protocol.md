# Zero-cost architecture-proxy protocol

## Scope

The formal SpatialFocus architecture-proxy experiment is **pre-SFT only**.
It evaluates the common pretrained VLM/CUT3R backbone plus each candidate's
exact pre-SFT injection architecture and initialization, on a shared fixed
supervised calibration minibatch.  Its target comparison is the known
post-SFT VSI-Bench architecture ranking.

`post-SFT` in that target ranking describes the downstream outcome; it does
**not** authorize evaluating proxy scores on trained SFT checkpoints.

## Prohibited substitution

Do not load `adapter_model.bin`, `non_lora_trainables.bin`, or another trained
post-SFT candidate checkpoint as a substitute for a missing pre-SFT candidate
initialization.  A read-only backward pass avoids further updates, but it still
measures a trained model and is not a pre-SFT architecture proxy.  Such a
result must not be pooled with, correlated as, or reported as the formal
pre-SFT proxy comparison.

If an architecture has no checkpoint-exact pre-SFT construction/initialization
artifact, record it as unavailable rather than redesigning it or falling back
to post-SFT weights.

## Post-SFT diagnostics

A score computed from an already trained checkpoint may be retained only as a
separately labelled checkpoint diagnostic.  It has no role in architecture
selection by the formal pre-SFT zero-cost proxy study and must not be presented
as evidence for that study.

## Immutable experiment conditions

- Use the exact candidate architectures in the relevant pre-SFT depth-probe
  study.
- Keep the pretrained shared backbone fixed.
- Use identical calibration minibatch(es) across candidates.
- Do not construct an optimizer or update parameters.
- Report whole-model scores only when the backward pass is safe; always label
  candidate-specific scores by the parameters that differ by architecture.
