# Event-Level Redesign

This document updates the original method with the new route:

- preprocessing stays unchanged;
- the main redesign focuses on event-level encoding and a stronger teacher;
- detection becomes event-level masked reconstruction instead of view-token reconstruction.

## Step1: Event-Level Multi-View Encoding

The original three-view split remains:

- `process`
- `file`
- `network`

What changes is the representation granularity. Instead of turning a whole view window directly into one vector, we first encode each event with its full semantic fields:

- event type
- operation
- fine-grained subtype
- main object key
- text-like fields such as command line, payload, path, and IP/port context
- field-availability pattern
- timestamp and local order
- host id

Each `host x 5min` window is then processed as an ordered event sequence.

The new Step1 path is:

1. Build event embeddings from complete event fields.
2. Compute lightweight per-view window statistics.
3. Feed the statistics into a router to predict subspace weights.
4. Form an alignment operator from shared bases.
5. Map event embeddings from different views into one shared event space.
6. Run a Transformer over the ordered event sequence.
7. Produce contextualized event embeddings and a pooled window embedding.

## Step2: Stronger Teacher

The teacher is upgraded from a fixed MLP encoder to a configurable backbone.

Supported backbones:

- `mlp`
- `transformer`

Recommended migration path:

1. keep the current MLP teacher for compatibility checks;
2. pre-adapt a stronger teacher on the first portion of global benign events;
3. switch student distillation to the stronger teacher outputs after sanity checks.

## Step3: Event-Level Masked Reconstruction

The original deep detector reconstructed four high-level tokens. The redesign changes the unit of masking and scoring from view tokens to events.

Training:

1. sort events by `host -> timestamp -> view`;
2. randomly mask valid events within the sequence;
3. reconstruct masked events from surrounding context;
4. optimize masked reconstruction loss with a light visible-token regularizer.

Inference:

1. explicitly mask events, preferably leave-one-out per event;
2. compute reconstruction loss for each event;
3. aggregate the highest event losses into a window anomaly score;
4. mark the suspicious `host x 5min` window and retain the most suspicious event/view as evidence.

## Code Mapping

New code added in this redesign:

- `src/optc_uras/models/event_encoder.py`
- `src/optc_uras/models/event_detector.py`
- `scripts/train_event_detector.py`

Teacher upgrade:

- `src/optc_uras/models/teacher.py`

These changes are additive. They do not remove the existing preprocessing or the old detector path, so we can compare old and new routes during migration.
